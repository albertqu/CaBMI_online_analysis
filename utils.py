counter_int = 60


def second_to_hmt(sec):
    h, sec = sec // 3600, sec % 3600
    m, sec = sec // 60, sec % 60
    return h, m, sec
