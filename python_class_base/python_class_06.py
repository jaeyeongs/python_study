# 연습문제
# 다음의 조건을 만족하는 Point라는 클래스를 작성
# Point 클래스는 생성자를 통해 (x, y) 좌표를 입력받는다.
# setx(x), sety(y) 메서드를 통해 x 좌표와 y 좌표를 따로 입력받을 수도 있다.
# get() 메서드를 호출하면 튜플로 구성된 (x, y) 좌표를 반환한다.
# move(dx, dy) 메서드는 현재 좌표를 dx, dy만큼 이동시킨다.
# 모든 메서드는 인스턴스 메서드다.

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def setx(self, x):
        self.x = x
    def sety(self, y):
        self.y = y
    def get(self):
            return (self.x, self.y)
    def move(self, dx, dy):
            self.x += dx
            self.y += dy

# a라는 Point 클래스 인스턴스 생성하고 x = 3, y = 3 할당
a = Point(3,3)
a.get()

# setx 매서드와 sety 매서드에 각각 x값 y값 할당
a.setx(4)
a.sety(2)
a.get()

# move 메서드를 이용해 dx는 -4만큼, dy는 -2만큼 할당하여 좌표 이동 후 확인
a.move(-4, -2)
a.get()
