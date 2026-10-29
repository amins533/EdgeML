import cv2
import numpy as np

class PlayCard:
    def __init__(self, width = 1200, height = 533):
        self.width = width
        self.height = height

        self.grass_color = (0, 128, 0)
        self.line_color = (255, 255, 255)
        self.color_touchdown = (220, 120, 60)
        # self.color_touchdown = (200, 100, 60)

        self.field = np.full((self.height, self.width, 3), self.grass_color, dtype = np.uint8)
        self.draw_base_field()

    def draw_base_field(self):
        yard_px = self.width / 120

        touch_down_width = int (10 * yard_px)
        cv2.rectangle(self.field, (0, 0), (touch_down_width, self.height), self.color_touchdown, -1)
        cv2.rectangle(self.field, (self.width - touch_down_width, 0), (self.width, self.height), self.color_touchdown, -1)

        for yard in range(10, 120, 10):
            x = int(yard * yard_px)
            cv2.line(self.field, (x, 0), (x, self.height), self.line_color, 1)

            if 10 <= yard <= 110:
                yard_number = yard if yard <= 50 else 120 - yard

                cv2.putText(self.field, str(yard_number), (x - 15, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.line_color, 2)
                cv2.putText(self.field, str(yard_number), (x - 15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.line_color, 2)
            
    def simulate_player_route(self):
        route_coor = [(200, 400), (300, 400), (400, 300), (500, 200)]

        for i in range(len(route_coor) - 1):
            pt1 = route_coor[i]
            pt2 = route_coor[i + 1]

            cv2.line(self.field, pt1, pt2, (255, 0, 0), 3)
            cv2.circle(self.field, pt1, 5, (0, 0, 255), -1)

    def show(self):
        cv2.imshow("Play Card", self.field)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    play_card = PlayCard()
    play_card.simulate_player_route()
    play_card.show()