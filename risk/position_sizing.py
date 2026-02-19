

def position_size(capital, risk_percent, atr):

    risk_amount = capital * risk_percent

    stop_distance = atr * 2

    if stop_distance <= 0:
        return 0

    size = risk_amount / stop_distance

    return size
