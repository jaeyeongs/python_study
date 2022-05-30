# BusineesCard라는 클래스에 메서드 추가하기
# 메스더 추가는 def로 정의하고 set_info 라는 메서드를 추가함
# set_info 라는 메서드에는 self, name, email, addr 네 개의 인자를 받음
# name, email, addr은 사용자로부터 입력받음
# self는 클래스 내부에 정의된 함수인 매서드의 첫 번째 인자는 반드시 self
class BusinessCard:
    def set_info(self, name, email, addr):
            self.name = name
            self.email = email
            self.addr = addr
    def print_info(self):
            print("--------------------")
            print("Name: ", self.name)
            print("E-mail: ", self.email)
            print("Address: ", self.addr)
            print("--------------------")

# 인스턴스 지정
member1 = BusinessCard()
member1.set_info("Yuna Kim", "yunakim@naver.com", "Seoul")

# 해당 메서드를 호출해 값을 출력
member1.print_info()