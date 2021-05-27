from . import space_00, space_01, space_02, space_03, space_10, space_11, space_12, space_13

spaces = {
    "00": space_00.functions,
    "01": space_01.functions,
    "02": space_02.functions,
    "03": space_03.functions,
    "10": space_10.functions,
    "11": space_11.functions,
    "12": space_12.functions,
    "13": space_13.functions
}

ns_branches = {
    "00": space_00.n_branches,
    "01": space_01.n_branches,
    "02": space_02.n_branches,
    "03": space_03.n_branches,
    "10": space_10.n_branches,
    "11": space_11.n_branches,
    "12": space_12.n_branches,
    "13": space_13.n_branches
}
