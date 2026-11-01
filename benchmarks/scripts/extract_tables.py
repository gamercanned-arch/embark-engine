with open(r'c:\Users\abhik\Documents\antigravity\silly-pascal\llama.cpp\ggml\src\ggml-common.h', 'r') as f:
    content = f.read()

# Extract iq2s_grid
start2 = content.find("GGML_TABLE_BEGIN(uint64_t, iq2s_grid, 1024)")
end2 = content.find("GGML_TABLE_END()", start2)
iq2s_body = content[start2:end2]
iq2s_body = iq2s_body.replace("GGML_TABLE_BEGIN(uint64_t, iq2s_grid, 1024)", "alignas(64) static const uint64_t iq2s_grid[1024] = {") + "\n};\n"

# Extract iq3s_grid
start3 = content.find("GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)")
end3 = content.find("GGML_TABLE_END()", start3)
iq3s_body = content[start3:end3]
iq3s_body = iq3s_body.replace("GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)", "alignas(64) static const uint32_t iq3s_grid[512] = {") + "\n};\n"

out_path = r'c:\Users\abhik\Documents\antigravity\silly-pascal\src\zen3_engine\zen3_iq_tables.h'
with open(out_path, 'w') as f:
    f.write("#pragma once\n#include <cstdint>\n\n")
    f.write(iq2s_body)
    f.write("\n")
    f.write(iq3s_body)

print(f"Generated {out_path} successfully!")
