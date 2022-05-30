# __init__()
class MyClass:
    def __init__(self):
            print("객체가 생성되었습니다.")

inst1 = MyClass()

class BusinessCard:
    def __init__(self, name, email, addr):
            self.name = name
            self.email = email
            self.addr = addr
    def print_info(self):
            print("--------------------")
            print("Name: ", self.name)
            print("E-mail: ", self.email)
            print("Address: ", self.addr)
            print("--------------------")

# 인스턴스를 생성할 때 3개의 인자를 전달하지 않으면 오류가 발생
# member1 = BusinessCard()

member1 = BusinessCard("Kangsan Lee", "kangsan.lee", "USA")
member1.print_info()