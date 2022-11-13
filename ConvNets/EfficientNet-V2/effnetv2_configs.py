
def get_config(config_name : str) -> list:
    """ Creates the configuration based on the config_name

    Args:
        config_name - there are 3 available configurations : Base, S, M, L, XL
        
    Return:
        list of updated values for each stage based on the config_name
    """

    width, depth = 1, 1

    assert config_name in ["Base", "S", "M", "L", "XL"], "Wrong configuration name."
    
    return configs[config_name]


"""EfficientNet V2 configs

stage string: 
    r - number of layers
    k - kernel size
    s - stride
    e - expansion ratio
    i - input channels
    o - output channels
    c - conv3x3?
    se(x) - squeeze and excitation block + (x) is ratio


"""
base_config = [  # The baseline config for v2 models.
    'r1_k3_s1_e1_i32_o16_c1',
    'r2_k3_s2_e4_i16_o32_c1',
    'r2_k3_s2_e4_i32_o48_c1',
    'r3_k3_s2_e4_i48_o96_se0.25',
    'r5_k3_s1_e6_i96_o112_se0.25',
    'r8_k3_s2_e6_i112_o192_se0.25',
]


s_config = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]


m_config = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',
]


l_config = [  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',
]

xl_config = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',
]

configs = {
    "Base" : base_config,
    "S" : s_config,
    "M" : m_config,
    "L" : l_config,
    "XL" : xl_config
}