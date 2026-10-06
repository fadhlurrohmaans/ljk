import random
import time
import os

class SandboxWorld:
    def __init__(self, width=15, height=10):
        self.width = width
        self.height = height
        # "." = Tanah Kosong, "T" = Pohon, "R" = Rumah
        self.grid = [['.' for _ in range(width)] for _ in range(height)]
        self.weather = "Cerah"
        # Menempatkan 3 penduduk secara acak
        self.agents = [{"x": random.randint(0, width-1), "y": random.randint(0, height-1)} for _ in range(3)]
        
        # Tumbuhkan beberapa pohon awal
        for _ in range(5):
            self.grid[random.randint(0, height-1)][random.randint(0, width-1)] = 'T'

    def god_manipulate(self, new_weather):
        self.weather = new_weather

    def update(self):
        # 1. Logika Alam (Cuaca mempengaruhi pertumbuhan)
        if self.weather == "Hujan" and random.random() < 0.4:
            rx, ry = random.randint(0, self.width-1), random.randint(0, self.height-1)
            if self.grid[ry][rx] == '.': 
                self.grid[ry][rx] = 'T' # Hujan menumbuhkan pohon baru

        # 2. Logika Agen (Penduduk beraktivitas)
        for agent in self.agents:
            # Penduduk bergerak acak mencari sumber daya
            agent['x'] = max(0, min(self.width-1, agent['x'] + random.choice([-1, 0, 1])))
            agent['y'] = max(0, min(self.height-1, agent['y'] + random.choice([-1, 0, 1])))

            # Jika penduduk menemukan Pohon (T), mereka membangun Rumah (R)
            if self.grid[agent['y']][agent['x']] == 'T':
                # Penduduk menolak bekerja jika ada Badai Petir
                if self.weather != "Badai Petir": 
                    self.grid[agent['y']][agent['x']] = 'R'

    def draw(self, hari):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"=== HARI {hari} | CUACA: {self.weather.upper()} ===")
        
        # Buat salinan grid untuk menampilkan penduduk (P) tanpa menimpa bangunan asli
        temp_grid = [row[:] for row in self.grid]
        for a in self.agents:
            if temp_grid[a['y']][a['x']] == '.':
                temp_grid[a['y']][a['x']] = 'P'

        for row in temp_grid:
            print(' '.join(row))
        print("\nLegenda: [.] Tanah  [T] Pohon  [R] Rumah  [P] Penduduk")

# Menjalankan Simulasi
dunia = SandboxWorld()
for hari in range(1, 11):
    # Simulasi Intervensi Dewa
    if hari == 3:
        dunia.god_manipulate("Hujan")
    elif hari == 6:
        dunia.god_manipulate("Badai Petir")
    elif hari == 8:
        dunia.god_manipulate("Cerah")

    dunia.update()
    dunia.draw(hari)
    time.sleep(1.5)
