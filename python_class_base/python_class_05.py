# self 이해하기
class Foo:
    def func1():
            print("function 1")
    def func2(self):
            print("function 2")

# func2에서는 self에 대한 값은 자동으로 넘겨주기 때문에 정상적으로 호출
f = Foo()
f.func2()

# 파이썬 메서드의 첫 번째 인자로 항상 인스턴스가 전달되기 때문에 발생하는 문제
# f.func1()

class Foo:
    def func1():
            print("function 1")

    def func2(self):
            print(id(self))
            print("function 2")

f = Foo()
id(f)

f.func2()