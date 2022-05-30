# 명함에 들어갈 정보
name = "kimyuna"
email = "yunakim@naver.com"
addr = "seoul"

# 명함을 출력하는 함수 생성
def print_business_card(name, email, addr):
    print("-------------------------")
    print("Name: %s" % name)
    print("E-mail: %s" % email)
    print("Office Address: %s" % addr)
    print("-------------------------")

# 명함 출력
print_business_card(name, email, addr)