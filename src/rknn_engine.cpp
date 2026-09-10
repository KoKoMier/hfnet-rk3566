#include "rknn_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

unsigned char* load_model_file(const char* filename, int* model_size) {
    FILE* fp = fopen(filename, "rb");
    if (fp == nullptr) {
        printf("Failed to open model file: %s\n", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* data = (unsigned char*)malloc(size);
    if (data == nullptr) {
        printf("Failed to allocate memory of size %d bytes\n", size);
        fclose(fp);
        return nullptr;
    }

    int read_bytes = fread(data, 1, size, fp);
    if (read_bytes != size) {
        printf("Failed to read model file: read %d, expected %d\n", read_bytes, size);
        free(data);
        fclose(fp);
        return nullptr;
    }

    fclose(fp);
    *model_size = size;
    return data;
}

void dump_tensor_attr(const rknn_tensor_attr* attr) {
    printf("  [Attr] index=%d, name=%s, n_dims=%d, dims=[", attr->index, attr->name, attr->n_dims);
    for (uint32_t i = 0; i < attr->n_dims; i++) {
        printf("%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
    }
    printf("], size=%d, stride=[w=%d, h=%d], format=%s, type=%s, scale=%.6f, zp=%d\n",
           attr->size, attr->w_stride, attr->h_stride, get_format_string(attr->fmt), get_type_string(attr->type), attr->scale, attr->zp);
}

RknnWorker::RknnWorker() = default;

RknnWorker::~RknnWorker() {
    release();
}

int RknnWorker::init(int worker_id, unsigned char* model_data, int model_len, bool dump_attrs) {
    worker_id_ = worker_id;
    int ret = rknn_init(&ctx_, model_data, model_len, 0, nullptr);
    if (ret != RKNN_SUCC) {
        printf("[RknnWorker %d] rknn_init failed! ret=%d\n", worker_id_, ret);
        return ret;
    }
    initialized_ = true;

    // 查询 IO 数量
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC) {
        printf("[RknnWorker %d] rknn_query IN_OUT_NUM failed! ret=%d\n", worker_id_, ret);
        return ret;
    }

    // 查询输入张量属性
    input_attrs_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; i++) {
        input_attrs_[i].index = i;
        rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
        if (dump_attrs) dump_tensor_attr(&(input_attrs_[i]));
    }

    // 查询输出张量属性
    output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; i++) {
        output_attrs_[i].index = i;
        rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        if (dump_attrs) dump_tensor_attr(&(output_attrs_[i]));
    }

    // 解析输入尺寸 (支持 NCHW / NHWC)
    req_height_ = (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) ? input_attrs_[0].dims[2] : input_attrs_[0].dims[1];
    req_width_  = (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) ? input_attrs_[0].dims[3] : input_attrs_[0].dims[2];

    // 解析输出张量索引
    logits_idx_ = -1;
    desc_idx_ = -1;
    for (size_t i = 0; i < output_attrs_.size(); ++i) {
        if (strcmp(output_attrs_[i].name, "logits") == 0) logits_idx_ = i;
        else if (strcmp(output_attrs_[i].name, "local_descriptor_map") == 0) desc_idx_ = i;
    }

    logits_is_nchw_ = (logits_idx_ >= 0) ? (output_attrs_[logits_idx_].dims[1] == 65) : false;
    desc_is_nchw_   = (desc_idx_ >= 0) ? (output_attrs_[desc_idx_].dims[1] == 256) : false;
    logits_scale_   = (logits_idx_ >= 0) ? output_attrs_[logits_idx_].scale : 1.0f;
    logits_zp_      = (logits_idx_ >= 0) ? output_attrs_[logits_idx_].zp : 0;

    // 创建零拷贝物理内存
    input_attrs_[0].type = RKNN_TENSOR_UINT8;
    input_mem_ = rknn_create_mem(ctx_, input_attrs_[0].size_with_stride);
    rknn_set_io_mem(ctx_, input_mem_, &input_attrs_[0]);

    output_mems_.resize(io_num_.n_output, nullptr);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        output_attrs_[i].type = RKNN_TENSOR_INT8;
        output_mems_[i] = rknn_create_mem(ctx_, output_attrs_[i].n_elems * sizeof(int8_t));
        rknn_set_io_mem(ctx_, output_mems_[i], &output_attrs_[i]);
    }

    return RKNN_SUCC;
}

int RknnWorker::run_inference(const uint8_t* resized_y,
                              std::vector<int8_t>& out_logits,
                              std::vector<int8_t>& out_desc,
                              double& npu_time_ms) {
    if (!initialized_) return -1;

    int stride = input_attrs_[0].w_stride;

    // 填充零拷贝输入物理内存 (处理 w_stride 对齐)
    if (stride == req_width_ || stride == 0) {
        memcpy(input_mem_->virt_addr, resized_y, req_width_ * req_height_);
    } else {
        const uint8_t* src = resized_y;
        uint8_t* dst = (uint8_t*)input_mem_->virt_addr;
        for (int h = 0; h < req_height_; ++h) {
            memcpy(dst, src, req_width_);
            src += req_width_;
            dst += stride;
        }
    }

    // 硬件 NPU 推理
    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = rknn_run(ctx_, nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    npu_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (ret != RKNN_SUCC) {
        printf("[RknnWorker %d] rknn_run failed! ret=%d\n", worker_id_, ret);
        return ret;
    }

    // 提取 INT8 输出
    if (logits_idx_ >= 0) {
        int size = output_attrs_[logits_idx_].n_elems;
        out_logits.resize(size);
        memcpy(out_logits.data(), output_mems_[logits_idx_]->virt_addr, size);
    }
    if (desc_idx_ >= 0) {
        int size = output_attrs_[desc_idx_].n_elems;
        out_desc.resize(size);
        memcpy(out_desc.data(), output_mems_[desc_idx_]->virt_addr, size);
    }

    return RKNN_SUCC;
}

void RknnWorker::release() {
    if (!initialized_) return;

    if (input_mem_) {
        rknn_destroy_mem(ctx_, input_mem_);
        input_mem_ = nullptr;
    }
    for (auto mem : output_mems_) {
        if (mem) rknn_destroy_mem(ctx_, mem);
    }
    output_mems_.clear();

    if (ctx_ > 0) {
        rknn_destroy(ctx_);
        ctx_ = 0;
    }
    initialized_ = false;
}
