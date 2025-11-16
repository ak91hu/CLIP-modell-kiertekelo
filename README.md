# CLIP kiértékelő

## 1. Leírás

Ez a Python script az OpenAI CLIP modelljének (`openai/clip-vit-base-patch32`) teljesítményét értékeli ki egy "zero-shot" képfelismerési feladaton. A script egy mappában található képeket dolgoz fel és egy előre megadott szabályrendszer alapján kiértékeli a modell pontosságát.

A script három fő kérdésre ad választ:
1.  **Bizonytalan:** Melyik táblatípusnál volt a modell legtöbbször bizonytalan (a legjobb tippje 50% alatti valószínűséget kapott)?
2.  **Téves:** Melyik táblatípust sorolta be a legtöbbször tévesen (magabiztosan, >50%-kal, de rossz kategóriába)?
3.  **Mindig helyes:** Melyik az a táblatípus, amit a modell minden egyes alkalommal helyesen és magabiztosan ismert fel?

## 2. Telepítés

### Előfeltételek
* Python 3.6+
* A képeket tartalmazó `data` mappa

### Függőségek telepítése
A script futtatásához telepítened kell a Hugging Face `transformers` könyvtárat, a `torch`-ot és a `Pillow` (PIL) képkezelő könyvtárat.

```bash
pip install transformers torch pillow
