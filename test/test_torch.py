from scipy.optimize import linprog

# Nhập số lượng biến và các hệ số ràng buộc
n, A, B, C = map(int, input().split())

# Nhập dữ liệu
costs = list(map(int, input().split()))
waters = list(map(int, input().split()))
profits = list(map(int, input().split()))

# Hệ số hàm mục tiêu (tối đa hóa profits => đổi dấu để tối thiểu hóa)
c = [-x for x in profits]

# Hệ số ràng buộc (Ax <= b)
bieuthuc_A = [
    costs,  # Tổng chi phí ≤ B
    waters,  # Tổng lượng nước ≤ C
    [1] * n,  # Tổng số lượng sản phẩm ≤ A
]
bieuthuc_b = [B, C, A]

# Giới hạn biến không âm
x_bounds = [(0, None)] * n

# Giải bài toán tối ưu hóa tuyến tính
result = linprog(c, A_ub=bieuthuc_A, b_ub=bieuthuc_b, bounds=x_bounds, method="highs")

# Xuất kết quả
if result.success:
    print(int(-result.fun))  # Đảo dấu để lấy giá trị lớn nhất

    for i in result.x:
        print(
            int(i)
        )  # Chuyển về số nguyên (vì linprog có thể trả về số thực rất gần số nguyên)
else:
    print("No optimal solution found")
