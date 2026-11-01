with open('stable-diffusion.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    if (backend_ctx->gpu_family == ADRENO) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }'''

repl = '''    if (backend_ctx->gpu_family == ADRENO || backend_ctx->gpu_family == AMD) {
        sgs = 64;
    } else if (backend_ctx->gpu_family == INTEL) {
        sgs = 32;
    } else {
        GGML_ASSERT(false && "Unsupported GPU");
    }'''

count = text.count(target)
print('Found instances:', count)
assert count == 4, f'Expected 4, got {count}'
text = text.replace(target, repl)

target2 = '''    } else if (backend_ctx->gpu_family == ADRENO) {
        nth0 = 64;
    }'''
repl2 = '''    } else if (backend_ctx->gpu_family == ADRENO || backend_ctx->gpu_family == AMD) {
        nth0 = 64;
    }'''
count2 = text.count(target2)
print('Found instances for nth0:', count2)
text = text.replace(target2, repl2)

with open('stable-diffusion.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp', 'w', encoding='utf-8') as f:
    f.write(text)
print('Successfully patched all instances!')
