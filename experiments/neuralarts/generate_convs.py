def twin(val):
    if type(val) != tuple and val != None:
        val = (val, val)
    return val


class ConvSettings:

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 dilation=1,
                 stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride

    def compute_internal(self):
        return 2 * self.padding - self.dilation * (self.kernel_size - 1)

    def compute_output(self, I):
        output = (I + self.compute_internal() - 1) / self.stride + 1
        return int(output)

    def copy(self, kernel_size, padding, dilation):
        return ConvSettings(self.in_channels, self.out_channels, kernel_size,
                            padding, dilation, self.stride)

    def is_complete_equivalent(self, other):
        for (self_prop, other_prop) in [
            (self.in_channels, other.in_channels),
            (self.out_channels, other.out_channels),
            (self.kernel_size, other.kernel_size),
            (self.padding, other.padding),
            (self.dilation, other.dilation),
            (self.stride, other.stride),
        ]:
            if self_prop != other_prop:
                return False

        return True

    def is_type_equivalent(self, other):
        if self.stride != other.stride:
            return False

        if self.compute_internal() != other.compute_internal():
            return False

        return True

    def is_instant_equivalent(self, other, I):
        return self.compute_output(I) == other.compute_output(I)

    def generate_settings_type_eq(self,
                                  k_limit=None,
                                  p_limit=None,
                                  d_limit=None):
        settings = []
        if [k_limit, p_limit, d_limit].count(None) != 1:
            print("ERROR: UNBOUNDED HYPERPARAMETERS")
            return settings

        def add_settings(k, p, d):
            new = self.copy(int(k), p, d)
            if self.is_type_equivalent(new):
                settings.append(new)

        k_limit, p_limit, d_limit = twin(k_limit), twin(p_limit), twin(d_limit)
        if k_limit is None:
            for p in range(p_limit[0], p_limit[1] + 1):
                for d in range(d_limit[0], d_limit[1] + 1):
                    k = (2 * p - self.compute_internal()) / d + 1
                    add_settings(k, p, d)
        elif p_limit is None:
            for k in range(k_limit[0], k_limit[1] + 1):
                for d in range(d_limit[0], d_limit[1] + 1):
                    p = int(d * (k - 1) / 2)
                    add_settings(k, p, d)
        elif d_limit is None:
            for k in range(k_limit[0], k_limit[1] + 1):
                for p in range(p_limit[0], p_limit[1] + 1):
                    d = int((2 * p) / (k - 1))
                    add_settings(k, p, d)

        return settings


if __name__ == "__main__":
    settings = ConvSettings(0, 0, 3, 1, 1, 1)

    ranges = {
        "kd04": ((1, 4), None, (1, 4)),
        "kd08": ((1, 8), None, (1, 8)),
        "kd12": ((1, 12), None, (1, 12)),
        "kd16": ((1, 16), None, (1, 16)),
        "kp04": ((2, 4), (1, 4), None),
        "kp08": ((2, 8), (1, 8), None),
        "kp12": ((2, 12), (1, 12), None),
        "kp16": ((2, 16), (1, 16), None),
        "pd04": (None, (1, 4), (1, 4)),
        "pd08": (None, (1, 8), (1, 8)),
        "pd12": (None, (1, 12), (1, 12)),
        "pd16": (None, (1, 16), (1, 16)),
    }
    for (name, (K, P, D)) in ranges.items():
        new_settings = settings.generate_settings_type_eq(K, P, D)
        content = ""

        for ns in new_settings:
            content += f"({ns.kernel_size},{ns.padding},{ns.dilation})\n"

        f = open(f"{name}.txt", "w")
        f.write(content)
        f.close()
