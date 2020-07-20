def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def calc_bflops(model_defs, input_size = 608, input_channel = 3):
    total_flops = 0
    flops_list = []
    for (idx, layer) in enumerate(model_defs):
        if layer['type'] != 'convolutional':
            if layer['type'] == 'upsample':
                input_size *= 2
            continue
        # if Downsample
        if layer['stride'] == '2':
            output_size = float(input_size) // 2
        else:
            output_size = float(input_size)
        # output channel
        kernel_channel = float(layer['filters'])
        # kernel size
        kernel_size = int(layer['size'])
        # calculate layer flops and append to list
        layer_flops = 2 * kernel_channel * kernel_size * kernel_size * output_size * output_size * input_channel
        print(f"(2 * {kernel_channel} * {kernel_size} * {kernel_size}) * {output_size} * {output_size} * {input_channel}")
        total_flops += layer_flops
        flops_list.append(layer_flops)
        # update input channel and input size
        input_channel = kernel_channel
        input_size = output_size
        print(f'BFLOPs for Layer {idx}: {layer_flops / 1e9}\n')
    print(f'total BFLOPS is {total_flops / 1e9}')
    return flops_list

if __name__ == '__main__':
    config_path = './yolov4.cfg'
    module_defs = parse_model_config(config_path)
    calc_bflops(module_defs, input_size=608)