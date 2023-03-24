#ifndef __YOLO_KERNEL_H
#define __YOLO_KERNEL_H

#include "utils.h"

void affine_project(float *matrix, float x, float y, float *ox, float *oy);

void decode_kernel_invoker(float *predict, int num_bboxes,
                        int num_classes, int output_cdim,
                        float confidence_threshold,
                        float nms_threshold,
                        float *invert_affine_matrix, float *parray,
                        Type type,
                        cudaStream_t stream);

void warp_affine_bilinear_and_normalize_plane(
            uint8_t *src, int src_line_size, int src_width, int src_height, float *dst,
            int dst_width, int dst_height, float *matrix_2_3, uint8_t const_value,
            const Norm &norm, cudaStream_t stream);

void decode_single_mask(float left, float top, float right, float bottom,
                        float *mask_weights, float *mask_predict,
                        int mask_width, int mask_height,
                        unsigned char *mask_out, int mask_dim,
                        int out_width, int out_height,
                        cudaStream_t stream);

#endif