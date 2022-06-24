# 4를 0으로 나누면 ZeroDivisionError 오류가 발생
# 4 / 0

# 오류 처리를 위한 try, except 기본 구조
# except [발생 오류 [as 오류 메시지 변수]]:

# 1. try, except만 쓰는 방법
#
# try:
#     ...
# except:
#     ...
#
# 2. 발생 오류만 포함한 except문
#
# try:
#     ...
# except 발생 오류:
#     ...
#
# 3. 발생 오류와 오류 메시지 변수까지 포함한 except문
#
# try:
#     ...
# except 발생 오류 as 오류 메시지 변수:
#     ...

try:
    4 / 0
except ZeroDivisionError as e:
    print(e)


# 여러개의 오류처리하기
# try문 안에서 여러 개의 오류를 처리하기 위해 다음 구문을 사용한다.
#
# try:
#     ...
# except 발생 오류1:
#    ...
# except 발생 오류2:
#    ...
try:
    a = [1,2]
    print(a[3])
    4/0
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
except IndexError:
    print("인덱싱 할 수 없습니다.")

try:
    a = [1,2]
    print(a[3])
    4/0
except ZeroDivisionError as e:
    print(e)
except IndexError as e:
    print(e)

# 2개 이상의 오류를 동일하게 처리하기 위해서는 다음과 같이 괄호를 사용하여 함께 묶어 처리
try:
    a = [1,2]
    print(a[3])
    4/0
except (ZeroDivisionError, IndexError) as e:
    print(e)