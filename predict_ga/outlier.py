def find_outliers(data, window_size=16, threshold=5):
    outliers = []
    n = len(data)
    for i in range(n - window_size + 1):
        window = data[i:i + window_size]
        mean = sum(window) / window_size
        std_dev = (sum((x - mean) ** 2 for x in window) / window_size) ** 0.5

        for j in range(window_size):
            if abs(window[j] - mean) > threshold * std_dev:
                outliers.append((i + j, window[j]))  # 记录位置和值
    return outliers


if __name__ == '__main__':
    data = [
        30.078259, 59.261530, 64.235910, 87.323061, 92.766544, 159.996517, 139.134591, 159.664855,
        123.895778, 176.171504, 178.664908, 172.474934, 135.508575, 119.606860, 139.197889, 147.125858,
        261.857124, 164.381266, 167.839050, 144.200264, 192.188654, 141.744063, 207.583245, 194.615752,
        190.613509, 204.038047, 181.601029, 179.865778, 190.425726, 176.115660, 98.204314, 219.852111,
        235.043615, 228.390106, 197.558470, 186.708997, 215.843852, 247.550686, 240.808298, 293.583536,
        191.118873, 216.796485, 240.628267, 296.684704, 249.878941, 260.642052, 244.858077, 301.729323,
        414.618367, 299.748288, 344.701058, 383.136943, 235.846491
    ]
    print(find_outliers(data,window_size=16, threshold=3))