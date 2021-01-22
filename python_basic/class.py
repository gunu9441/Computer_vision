class Country:
    """Super Class"""

    def __init__(self):
        print("hello")
        self.name = '국가명'
        self.population = '인구'
        self.capital = '수도'

    def show(self):
        print('국가 클래스의 메소드입니다.')
        print("""{} {} {} """.format(self.name, self.population, self.capital))


class Korea(Country):
    """Sub Class"""

    def __init__(self, name):
        print("hi")
        super().__init__()  # super class를 초기화 시켜주기위해 반드시 적어주어야함.
        self.name = name

    def show_name(self):
        print('국가 이름은 : ', self.name)


a = Korea('대한민국')
a.show()
print(a.population)
