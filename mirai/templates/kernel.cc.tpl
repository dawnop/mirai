#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"


static inline void gpuAssert(CUresult code, const char *file, int line) {
    if (code != CUDA_SUCCESS)
    {
        const char *str;
        cuGetErrorString(code, &str);
        char err[1024] = {0};
        strcat(err, str);
        std::cerr << "Error Code: " << code
                  << "\n\tMessage: " << err
                  << "\n\tAt " << file << ":" << line << std::endl;
        std::abort();
    }
}

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

void process_cufunc(CUfunction fun, int shared, int device) {
    int shared_optin;
    CUDA_CHECK(cuDeviceGetAttribute(
        &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        device));
    if (shared > 49152 && shared_optin > 49152) {
        CUDA_CHECK(cuFuncSetCacheConfig(fun, CU_FUNC_CACHE_PREFER_SHARED));
        int shared_total, shared_static;
        CUDA_CHECK(cuDeviceGetAttribute(
            &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            device));
        CUDA_CHECK(cuFuncGetAttribute(&shared_static,
                                      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fun));
        CUDA_CHECK(
            cuFuncSetAttribute(fun, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               shared_optin - shared_static));
    }
}

std::string read_ptx(const std::string &filepath) {
    std::ifstream fs(filepath);
    if (!fs.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::stringstream buffer;
    buffer << fs.rdbuf();
    return buffer.str();
}

namespace tensorflow
{
    struct LaunchConfig {
        std::string func_name;
        int32_t shared_memory;
        int32_t num_warps;
        std::map<std::string, int> blocks;
    };

    void read_config(OpKernelConstruction *ctx, std::string &ptx_conf_path, LaunchConfig &config) {
        std::ifstream fin(ptx_conf_path);
        OP_REQUIRES(ctx, fin.is_open(),
                    tensorflow::errors::InvalidArgument("Failed to open config file: ", ptx_conf_path));

        std::string header, values;
        OP_REQUIRES(ctx, std::getline(fin, header), 
                    tensorflow::errors::InvalidArgument("Config file format error: missing header line in ", ptx_conf_path));
        OP_REQUIRES(ctx, std::getline(fin, values), 
                    tensorflow::errors::InvalidArgument("Config file format error: missing values line in ", ptx_conf_path));

        std::istringstream hss(header), vss(values);
        std::vector<std::string> keys;
        std::vector<std::string> vals;
        std::string token;

        while (hss >> token) keys.push_back(token);
        while (vss >> token) vals.push_back(token);

        OP_REQUIRES(ctx, vals.size() >= 3,
                    tensorflow::errors::InvalidArgument("Config file values error: need at least 3 values in ", ptx_conf_path));
        OP_REQUIRES(ctx, keys.size() >= 3, 
                    tensorflow::errors::InvalidArgument("Config file header error: need at least 3 keys (func_name, shared, num_warps) in ", ptx_conf_path));
        config.func_name = vals[0];
        config.shared_memory = std::stoi(vals[1]);
        config.num_warps = std::stoi(vals[2]);
        for (size_t i = 3; i < keys.size(); ++i) {
            OP_REQUIRES(ctx, i < vals.size(),
                        tensorflow::errors::InvalidArgument("Config file values error: missing value for key ", keys[i], " in ", ptx_conf_path));
            config.blocks[keys[i]] = std::stoi(vals[i]);
        }
    }

    template <typename Tp>
    DataType get_type() {
        if (std::is_same<Tp, float>::value) {
            return DT_FLOAT;
        } else if (std::is_same<Tp, Eigen::half>::value) {
            return DT_HALF;
        } else if (std::is_same<Tp, tensorflow::bfloat16>::value) {
            return DT_BFLOAT16;
        } else {
            return DT_INVALID;
        }
    }

    void init_kernel(OpKernelConstruction *ctx, std::string asset_prefix, const std::string& module_name, LaunchConfig& config, CUfunction& cufunc) {
        auto ptx_conf_path = asset_prefix + module_name + "_meta.txt";
        read_config(ctx, ptx_conf_path, config);

        auto ptx_path = asset_prefix + module_name + ".ptx";
        auto ptx = read_ptx(ptx_path);
        CUmodule mod = nullptr;
        CUDA_CHECK(cuModuleLoadData(&mod, ptx.c_str()));
        CUDA_CHECK(cuModuleGetFunction(&cufunc, mod, config.func_name.c_str()));

        auto device = ctx->device()->tensorflow_gpu_device_info()->gpu_id;
        process_cufunc(cufunc, config.shared_memory, device);

        LOG(INFO) << "init kernel:" << module_name;
    }

    bool check_tensor_shape_flexible(const tensorflow::Tensor& tensor, const std::vector<int64_t>& expected_shape) {
        const tensorflow::TensorShape& actual_shape = tensor.shape();

        int i = 0, j = 0;
        int actual_dims = actual_shape.dims();
        int expected_dims = expected_shape.size();

        while (i < actual_dims && j < expected_dims) {
            int64_t actual_dim = actual_shape.dim_size(i);
            int64_t expected_dim = expected_shape[j];

            // Dimensions match, advance both pointers
            if (actual_dim == expected_dim) {
                ++i;
                ++j;
            } else {
                // Accumulate the smaller side
                int64_t actual_acc = actual_dim;
                int64_t expected_acc = expected_dim;
                int ii = i, jj = j;

                // Accumulate actual dims
                while (actual_acc < expected_acc && ++ii < actual_dims) {
                    actual_acc *= actual_shape.dim_size(ii);
                }
                // Accumulate expected dims
                while (expected_acc < actual_acc && ++jj < expected_dims) {
                    expected_acc *= expected_shape[jj];
                }

                // If accumulated products match, advance pointers
                if (actual_acc == expected_acc) {
                    i = ii + 1;
                    j = jj + 1;
                } else {
                    return false; // Cannot match
                }
            }
        }
        // Both traversed completely — shapes are compatible
        return i == actual_dims && j == expected_dims;
    }


    template <typename Tp>
    class {{op_name}} : public OpKernel {
    public:
        explicit {{op_name}}(OpKernelConstruction *ctx);
        ~{{op_name}}() = default;
        void Compute(OpKernelContext *ctx) override;

    private:
        std::map<std::string, LaunchConfig> launch_config_map_;
        std::map<std::string, CUfunction> cufunc_map_;
        std::vector<std::string> kernel_names_;
    };

    template <typename Tp>
    {{op_name}}<Tp>::{{op_name}}(OpKernelConstruction *ctx) : OpKernel(ctx) {
        std::string asset_prefix;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("asset_prefix", &asset_prefix));
        kernel_names_ = { {% for name in kernel_names -%}
        "{{ name }}",
        {%- endfor %} };
        for (auto& module_name : kernel_names_) {
            LaunchConfig& config = launch_config_map_[module_name];
            CUfunction& cufunc = cufunc_map_[module_name];
            init_kernel(ctx, asset_prefix, module_name, config, cufunc);
        }

    }

    template <typename Tp>
    void {{op_name}}<Tp>::Compute(OpKernelContext *ctx) {

        {% for arg in input_args -%}
        const Tensor& {{ arg.name }} = ctx->input({{ loop.index0 }});
        {% endfor %}

        {% for input_arg in input_args -%}
            {% for output_arg in output_args -%}
                {% if input_arg.name == output_arg.name -%}
        ctx->set_output({{ loop.index0 }}, {{ input_arg.name }});
                {%- endif %}
            {%- endfor %}
        {% endfor %}

        AllocatorAttributes alloc_attrs;
        alloc_attrs.set_gpu_compatible(true);

        auto *se_stream = ctx->op_device_context()->stream();
        CUstream custream = static_cast<CUstream>(se_stream->implementation()->GpuStreamHack());

        {% for input_arg in input_args -%}
        auto {{ input_arg.name }}_ptr = (CUdeviceptr)({{ input_arg.name }}.data());
        {% endfor %}

        {% for code in compute_codes %}
        {{ code }}
        {% endfor %}

        cudaStreamSynchronize(custream);
    }

    REGISTER_OP("{{op_name}}")
        .Attr("asset_prefix: string")
        {% for arg in input_args %}
        .Input("{{ arg.name }}: T")
        {%- endfor %}
        {% for out in output_args %}
        .Output("{{ out.name }}_out: T")
        {%- endfor %}
        .Attr("T: {float, half, bfloat16} = DT_FLOAT")
        .SetShapeFn([](shape_inference::InferenceContext* c) {
            {% for out in output_args %}
            shape_inference::ShapeHandle {{out.name}} = c->MakeShape({ {{ out.shape | join(', ') }} });
            c->set_output({{loop.index0}}, {{out.name}});
            {%- endfor %}

            return Status::OK();
        });


    REGISTER_KERNEL_BUILDER(
        Name("{{op_name}}")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
        {{op_name}}<float>);
        
    REGISTER_KERNEL_BUILDER(
        Name("{{op_name}}")
        .Device(DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T"),
        {{op_name}}<Eigen::half>);

    REGISTER_KERNEL_BUILDER(
        Name("{{op_name}}")
        .Device(DEVICE_GPU)
        .TypeConstraint<tensorflow::bfloat16>("T"),
        {{op_name}}<tensorflow::bfloat16>);

    template class {{op_name}}<float>;
    template class {{op_name}}<Eigen::half>;
    template class {{op_name}}<tensorflow::bfloat16>;

}