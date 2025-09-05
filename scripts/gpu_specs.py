#!/usr/bin/env python3

import torch
import pynvml

def main():
    arch_map = {
        '5.0': 'Maxwell', '5.2': 'Maxwell', '5.3': 'Maxwell',
        '6.0': 'Pascal', '6.1': 'Pascal', '6.2': 'Pascal',
        '7.0': 'Volta', '7.2': 'Volta', '7.5': 'Turing',
        '8.0': 'Ampere', '8.6': 'Ampere', '8.7': 'Ampere', '8.9': 'Ada',
        '9.0': 'Hopper'
    }

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    name_str = name if isinstance(name, str) else name.decode()
    cap = pynvml.nvmlDeviceGetCudaComputeCapability(h)

    d = torch.cuda.get_device_properties(0)
    cap_str = f'{cap[0]}.{cap[1]}'
    arch = arch_map.get(cap_str, f'Compute {cap_str}')
    max_blocks = 32 if cap[0] >= 6 else 16

    # PyTorch reports operational limits, get architectural maximums for occupancy
    pytorch_shared_mem = d.shared_memory_per_multiprocessor // 1024

    # Architectural maximums (for occupancy calculations)
    if cap[0] >= 9:  # Hopper
        arch_shared_mem = 228
    elif cap[0] == 8:  # Ampere/Ada
        arch_shared_mem = 128 if cap[1] == 9 else 164  # Ada=128KB, Ampere=164KB
    elif cap[0] == 7:  # Volta/Turing
        arch_shared_mem = 96 if cap[1] >= 5 else 80
    elif cap[0] == 6:  # Pascal
        arch_shared_mem = 96
    else:  # Maxwell and older
        arch_shared_mem = 64

    print(f'Device: {name_str}')
    print(f'Architecture: {arch}')
    print(f'Compute Cap: {cap_str}')
    print(f'SM Count: {d.multi_processor_count}')
    print(f'Registers/SM: {d.regs_per_multiprocessor:,}')
    print(f'Shared Mem/SM: {arch_shared_mem}KB (architectural max)')
    print(f'Shared Mem/SM: {pytorch_shared_mem}KB (operational limit)')
    print(f'Max Threads/SM: {d.max_threads_per_multi_processor:,}')
    print(f'Max Blocks/SM: {max_blocks}')

if __name__ == '__main__':
    main()
