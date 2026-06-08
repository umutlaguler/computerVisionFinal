# Receipt Scanner + OCR — Klasik Bilgisayarlı Görü Pipeline'ı

Eğri/açılı çekilmiş bir **market fişini** otomatik tespit edip perspektif
düzeltmesiyle düzleştiren ve metnini çıkaran bir sistem. Tamamı **klasik
bilgisayarlı görü** teknikleriyle kurulmuştur (derin öğrenme tabanlı belge
segmentasyonu kullanılmaz); OCR motoru olarak **Tesseract** kullanılır.

> Computer Vision dönem projesi. Veri seti: 20 İngilizce ABD fişi (Walmart,
> Whole Foods, Costco…) ve CVAT anotasyonları (belge poligonu + alan metinleri).

---

## Pipeline

```
Girdi foto
  → 1. Yeniden boyutlandırma         (cv2.resize)
  → 2. Gri tonlama                   (cv2.cvtColor)
  → 3. Gürültü azaltma               (cv2.bilateralFilter / GaussianBlur)
  → 4. Kontrast (CLAHE)              (cv2.createCLAHE)
  → 5a. Kenar tespiti                (cv2.Canny — rapor görseli)
  → 5b. Otsu segmentasyon + morfoloji(cv2.threshold+OTSU, morphologyEx)  ← tespit bunu kullanır
  → 6. Kontur bulma                  (cv2.findContours, contourArea)
  → 7. Dört köşe (poligon yaklaşımı) (cv2.arcLength, approxPolyDP)
  → 8. Köşe sıralama                 (NumPy toplam/fark)
  → 9. Perspektif dönüşümü/homografi (getPerspectiveTransform, warpPerspective)
  → 10. Adaptive threshold           (cv2.adaptiveThreshold)  ← "taranmış" görsel
  → 11. OCR                          (pytesseract — gri-büyütülmüş görüntüde)
```

### Tasarım notları (neden böyle?)
- **Tespit, Canny kenarları yerine Otsu segmentasyonu üzerinden yapılır.**
  Fişin yoğun iç metni Canny'de çok sayıda kopuk kenara bölünür ve dış sınır
  düşük kontrastta kapanmaz; oysa fiş, karedeki en büyük *parlak* bölgedir, bu
  yüzden Otsu eşik + büyük morfolojik kapama fişi tek bir dolu bloğa indirger.
  Canny aşaması yine de hesaplanır ve görselleştirilir (kenar tespiti kavramını
  raporda göstermek için).
- **OCR sert binary yerine gri tonlamada çalışır.** Tesseract kendi iç
  eşiklemesini (Otsu) yapar; sert adaptive-threshold binary bilgi sildiği için
  OCR'ı kötüleştirir. Deneysel olarak gri-büyütülmüş (1.5×) girdi en iyi sonucu
  verir. Adaptive threshold çıktısı yalnızca "taranmış belge" görseli olarak
  saklanır.

---

## Klasik CV kavramları (rapor/sunum için)

| Adım | Temel kavram |
|------|--------------|
| Gri + Gaussian/bilateral blur | Görüntü ön işleme, gürültü modeli, kenar-koruyan filtreleme |
| Canny | Gradyan tabanlı kenar tespiti (Sobel + NMS + hysteresis) |
| Otsu eşik | Bimodal histogram ayrımı ile küresel segmentasyon |
| Morfolojik kapama/açma | İkili morfoloji, yapısal eleman (structuring element) |
| Kontur + approxPolyDP | Sınır temsili; Douglas–Peucker poligon yaklaşımı |
| Perspektif dönüşümü | Projektif geometri; 4 nokta eşleşmesinden 3×3 homografi (8 DOF) |
| Adaptive threshold | Düzgün olmayan ışıkta yerel (lokal) eşikleme |

---

## Kurulum

```bash
# tesseract sistem binary'si (macOS)
brew install tesseract

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Kullanım

```bash
# Tek görsel — tüm ara aşamaları data/results/<isim>/ altına kaydeder
python main.py data/raw/images/0.jpg

# Toplu işleme
python main.py data/raw/images/*.jpg

# Rapor için tek-görsel etiketli montaj figürü
python make_figure.py data/raw/images/0.jpg

# Streamlit demo arayüzü
streamlit run app.py

# Sayısal değerlendirme (IoU + alan geri çağırma, ham vs pipeline)
python evaluate.py

# Perspektif düzeltmenin değeri — kontrollü sentetik ablasyon
python ablation_perspective.py
```

---

## Değerlendirme ve sonuçlar

Ground truth `data/raw/annotations.xml` içinden gelir; metni elle yazmaya gerek
yoktur.

**1. Belge tespiti (geometri).** Bulunan dörtgen ile ground-truth `receipt`
poligonu arasında IoU.
- IoU ≥ 0.70 başarı: **14/20 (%70)**, ortalama IoU **0.78**.
- Not: Ground truth çok-noktalı, kıvrımlı bir poligondur (parmağı dışlar); bizim
  çıktımız düz bir dörtgendir, dolayısıyla IoU'nun doğal bir üst sınırı vardır.

**2. OCR doğruluğu (alan geri çağırma).** Anotasyon yalnızca seçili alanları
(mağaza/ürün/tarih/toplam) işaretlediği, OCR ise tüm fişi okuduğu için global
CER/WER "insertion"larla şişer. Bunun yerine her GT alanı için OCR çıktısındaki
en benzer satırın benzerliği [0,1] ölçülür.
- Düz (gerçek) fişlerde: ham OCR **0.815** ≈ pipeline **0.810** — pipeline ham
  OCR'a *eşittir* ve ek olarak kırpılmış/düzleştirilmiş temiz tarama üretir.

**3. Perspektif ablasyonu (kontrollü).** Düz fişlere bilinen bir perspektif
bozulması uygulanıp (açılı çekim simülasyonu) ham OCR ile pipeline karşılaştırılır.
- Ortalama benzerlik: ham **0.553** → pipeline **0.678** (**+0.125**). Ham OCR'ın
  çöktüğü güçlü açılarda kazanç +0.6'ya kadar çıkar. **Homografi adımının değeri
  bu deneyde kanıtlanır.**

---

## Sınırlamalar (gelecek çalışmalar)
- **Düşük kontrastlı arka plan** (ör. bej kumaş üzerinde beyaz fiş, `14.jpg`):
  Otsu fişi arka planla birleştirir, tespit bozulur.
- **Kıvrık/buruşuk fişler**: Perspektif dönüşümü düzlemsellik varsayar; fiziksel
  kıvrımı düzeltemez.
- **Güçlü açı + zayıf segmentasyon**: ablasyonda bazı örneklerde tespit çöker.
- İyileştirme yönleri: manuel köşe seçme fallback'i, dewarping (kıvrım düzeltme),
  alan ayrıştırma (toplam/tarih regex), `tur+eng` çoklu dil.

---

## Proje yapısı

```
.
├── main.py                 # CLI: tek/çoklu görsel → ara aşamalar + text
├── app.py                  # Streamlit demo arayüzü
├── evaluate.py             # IoU + alan geri çağırma metrikleri
├── ablation_perspective.py # perspektif düzeltmenin kontrollü ablasyonu
├── make_figure.py          # rapor için etiketli montaj figürü
├── requirements.txt
├── src/
│   ├── preprocess.py       # gri, blur, CLAHE, Canny, Otsu segmentasyon
│   ├── detect.py           # kontur + 4 köşe / min-area fallback
│   ├── transform.py        # köşe sıralama + homografi warp
│   ├── postprocess.py      # adaptive threshold + OCR girdisi hazırlama
│   ├── ocr.py              # pytesseract sarmalayıcı
│   ├── pipeline.py         # uçtan uca akış (PipelineResult)
│   └── annotations.py      # CVAT annotations.xml parser
└── data/
    ├── raw/                # images/, boxes/, annotations.xml
    └── results/            # çıktılar (gitignore'da)
```
