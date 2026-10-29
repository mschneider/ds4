#include <stdio.h>

#include "../ds4_metal_device.h"

int main(void) {
    const struct {
        const char *name;
        int modern;
        int qwen_legacy;
        int qwen_legacy_moe;
        int m4;
    } cases[] = {
        {NULL, 0, 0, 0, 0},
        {"", 0, 0, 0, 0},
        {"Apple", 0, 0, 0, 0},
        {"Apple M", 0, 0, 0, 0},
        {"Apple M1", 0, 0, 0, 0},
        {"Apple M1 Pro", 0, 0, 0, 0},
        {"Apple M1 Max", 0, 1, 1, 0},
        {"Apple M1 Ultra", 0, 0, 0, 0},
        {"Apple M1 Maxfoo", 0, 0, 0, 0},
        {"Apple M1 Max Ultra", 0, 0, 0, 0},
        {"Apple M2", 0, 1, 1, 0},
        {"Apple M2 Pro", 0, 1, 1, 0},
        {"Apple M2 Max", 0, 1, 1, 0},
        {"Apple M2 Ultra", 0, 1, 1, 0},
        {"Apple M3", 0, 1, 1, 0},
        {"Apple M3 Pro", 0, 1, 1, 0},
        {"Apple M3 Max", 0, 1, 1, 0},
        {"Apple M3 Ultra", 0, 1, 0, 0},
        {"Apple M4", 0, 1, 1, 1},
        {"Apple M4 Pro", 0, 1, 1, 1},
        {"Apple M4 Max", 0, 1, 1, 1},
        {"Apple M4 Ultra", 0, 1, 1, 1},
        {"Apple M5", 1, 0, 0, 0},
        {"Apple M5 Pro", 1, 0, 0, 0},
        {"Apple M5 Max", 1, 0, 0, 0},
        {"Apple M5 Ultra", 1, 0, 0, 0},
        {"Apple M6", 1, 0, 0, 0},
        {"Apple M6 Pro", 1, 0, 0, 0},
        {"Apple M6 Max", 1, 0, 0, 0},
        {"Apple M6 Ultra", 1, 0, 0, 0},
        {"Apple M7", 0, 0, 0, 0},
        {"Apple M20", 0, 0, 0, 0},
        {"Apple M30 Max", 0, 0, 0, 0},
        {"Apple M40 Pro", 0, 0, 0, 0},
        {"Apple M50", 0, 0, 0, 0},
        {"Apple M60 Max", 0, 0, 0, 0},
        {"Apple M2foo", 0, 0, 0, 0},
        {"Apple M3foo", 0, 0, 0, 0},
        {"Apple M4foo", 0, 0, 0, 0},
        {"Apple M5x", 0, 0, 0, 0},
        {"Apple M6x", 0, 0, 0, 0},
        {"Apple A19", 0, 0, 0, 0},
        {"Apple A20", 0, 0, 0, 0},
        {"AMD M2", 0, 0, 0, 0},
        {"AMD M5", 0, 0, 0, 0},
        {"M2", 0, 0, 0, 0},
        {"M6", 0, 0, 0, 0},
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
        const int modern = ds4_metal_device_name_is_m5_or_m6(cases[i].name);
        const int legacy =
            ds4_metal_device_name_uses_qwen_legacy_tuning(cases[i].name);
        const int legacy_moe =
            ds4_metal_device_name_uses_qwen_legacy_moe_tuning(cases[i].name);
        const int m4 = ds4_metal_device_name_is_m4(cases[i].name);
        if (modern != cases[i].modern || legacy != cases[i].qwen_legacy ||
            legacy_moe != cases[i].qwen_legacy_moe || m4 != cases[i].m4) {
            fprintf(stderr, "Metal device policy: %s: expected modern/legacy/MoE/M4 "
                    "%d/%d/%d/%d, got %d/%d/%d/%d\n",
                    cases[i].name ? cases[i].name : "(null)",
                    cases[i].modern, cases[i].qwen_legacy,
                    cases[i].qwen_legacy_moe, cases[i].m4,
                    modern, legacy, legacy_moe, m4);
            return 1;
        }
    }
    puts("Metal device policy: Qwen M1 Max/M2-M4, M4 and M5/M6 boundaries passed");
    return 0;
}
