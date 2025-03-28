#include <stdio.h>
#include <mach/mach.h>
#include <unistd.h> // sleep 関数のために追加
#include <stdlib.h> // malloc, free のために追加
#include <string.h> // memset のために追加

#define FIFTY_MB (50 * 1024 * 1024) // 50MB

int main()
{
    // 50MB のメモリを確保
    char *memory = (char *)malloc(FIFTY_MB);
    if (memory == NULL)
    {
        printf("Failed to allocate memory\n");
        return 1;
    }

    // 確保したメモリに適当な値を書き込む
    memset(memory, 42, FIFTY_MB); // 42という値で埋める
    printf("Allocated 50MB of memory and filled with data\n");

    while (1) // 無限ループに変更
    {
        struct task_vm_info info;
        mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
        kern_return_t kr = task_info(mach_task_self(),
                                     TASK_VM_INFO,
                                     (task_info_t)&info,
                                     &count);
        if (kr == KERN_SUCCESS)
        {
            printf("phys_footprint: %llu bytes\n", info.phys_footprint);
            printf("resident_size:  %llu bytes\n", info.resident_size);
            printf("virtual_size:   %llu bytes\n", info.virtual_size);
            printf("------------------------------\n"); // 区切り線を追加
        }

        sleep(1); // 1秒間スリープ
    }

    // この行は実行されないが、念のため
    free(memory);
    return 0;
}
