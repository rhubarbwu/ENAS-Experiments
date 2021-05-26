from . import space_00, space_01, space_02, space_03, space_10

spaces = {
    "00": space_00.functions,
    "01": space_01.functions,
    "02": space_02.functions,
    "03": space_03.functions,
    "10": space_10.functions
}

ns_branches = {
    "00": space_00.n_branches,
    "01": space_01.n_branches,
    "02": space_02.n_branches,
    "03": space_03.n_branches,
    "10": space_10.n_branches
}
