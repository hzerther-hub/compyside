"""One-shot script: translate the 37 newly-added EN keys into all 15
languages and write the merged JSON.

Run from repo root:

    python tools/apply_translations.py

The translation table below is hand-curated for each language. Anything
that the table doesn't cover keeps the English string (i.e. falls through
to EN at lookup time).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "src" / "compositor" / "i18n_data.json"


# ---------------------------------------------------------------------------
# Per-language overrides for the 37 new keys (added in tier 1-3 + AI).
# Each entry maps EN key → translation. Missing keys are left to fall
# through to English.
# ---------------------------------------------------------------------------

TRANSLATIONS: dict[str, dict[str, str]] = {
    "zh_CN": {
        # Layers panel / inspector
        "Tools": "工具", "Blend": "混合", "Opacity": "不透明度",
        "Visible": "可见", "Name": "名称", "Transform": "变换",
        "X": "X", "Y": "Y", "Width": "宽度", "Height": "高度",
        "Rotation": "旋转", "Flip Horizontal": "水平翻转",
        "Flip Vertical": "垂直翻转", "origin": "原点",
        "size": "大小", "rotation": "旋转角度",
        "Mask": "蒙版", "Mask enabled": "启用蒙版",
        "Invert": "反相", "Feather…": "羽化…",
        "Clear": "清除", "Apply": "应用", "Edit Text…": "编辑文字…",
        # AI
        "AI Backend Settings": "AI 后端设置",
        "AI background removal applied (U²-Net)": "AI 抠图已应用(U²-Net)",
        "AI Select Subject applied (rembg)": "AI 主体选择已应用(rembg)",
        "rembg not installed": "未安装 rembg",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "安装 rembg 以使用 AI 抠图:\\n\\npip install rembg",
        # Status messages
        "Removed background on active layer": "已在当前图层移除背景",
        "Content-Aware Fill applied": "已应用内容感知填充",
        "Subject selected": "已选择主体",
        "Upscaled active layer to {w}×{h}": "当前图层已放大至 {w}×{h}",
        "Copied merged to clipboard": "已复制合并图像到剪贴板",
        "Canvas resized {a}×{b} → {c}×{d}":
            "画布已调整 {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "图像已调整至 {w}×{h}",
        "Trim: image is empty": "裁切:图像为空",
        "Trimmed {a}×{b} → {c}×{d}": "已裁切 {a}×{b} → {c}×{d}",
        "Saved {p}": "已保存 {p}",
    },
    "ja": {
        "Tools": "ツール", "Blend": "ブレンド", "Opacity": "不透明度",
        "Visible": "表示", "Name": "名前", "Transform": "変形",
        "X": "X", "Y": "Y", "Width": "幅", "Height": "高さ",
        "Rotation": "回転", "Flip Horizontal": "水平反転",
        "Flip Vertical": "垂直反転", "origin": "原点",
        "size": "サイズ", "rotation": "回転",
        "Mask": "マスク", "Mask enabled": "マスク有効",
        "Invert": "反転", "Feather…": "ぼかし…",
        "Clear": "クリア", "Apply": "適用", "Edit Text…": "テキストを編集…",
        "AI Backend Settings": "AI バックエンド設定",
        "AI background removal applied (U²-Net)": "AI 背景除去を適用 (U²-Net)",
        "AI Select Subject applied (rembg)": "AI 被写体選択を適用 (rembg)",
        "rembg not installed": "rembg がインストールされていません",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "AI 背景除去に rembg をインストール:\\n\\npip install rembg",
        "Removed background on active layer": "アクティブレイヤーの背景を除去しました",
        "Content-Aware Fill applied": "コンテンツに応じた塗りつぶしを適用",
        "Subject selected": "被写体を選択しました",
        "Upscaled active layer to {w}×{h}": "アクティブレイヤーを {w}×{h} に拡大しました",
        "Copied merged to clipboard": "統合画像をクリップボードにコピーしました",
        "Canvas resized {a}×{b} → {c}×{d}":
            "キャンバスを {a}×{b} → {c}×{d} にリサイズしました",
        "Image resized to {w}×{h}": "画像を {w}×{h} にリサイズしました",
        "Trim: image is empty": "トリム:画像が空です",
        "Trimmed {a}×{b} → {c}×{d}": "{a}×{b} → {c}×{d} にトリムしました",
        "Saved {p}": "{p} を保存しました",
    },
    "ko": {
        "Tools": "도구", "Blend": "블렌드", "Opacity": "불투명도",
        "Visible": "표시", "Name": "이름", "Transform": "변형",
        "X": "X", "Y": "Y", "Width": "너비", "Height": "높이",
        "Rotation": "회전", "Flip Horizontal": "수평 뒤집기",
        "Flip Vertical": "수직 뒤집기", "origin": "원점",
        "size": "크기", "rotation": "회전",
        "Mask": "마스크", "Mask enabled": "마스크 사용",
        "Invert": "반전", "Feather…": "페더링…",
        "Clear": "지우기", "Apply": "적용", "Edit Text…": "텍스트 편집…",
        "AI Backend Settings": "AI 백엔드 설정",
        "AI background removal applied (U²-Net)": "AI 배경 제거 적용됨 (U²-Net)",
        "AI Select Subject applied (rembg)": "AI 피사체 선택 적용됨 (rembg)",
        "rembg not installed": "rembg이 설치되지 않음",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "AI 배경 제거용 rembg 설치:\\n\\npip install rembg",
        "Removed background on active layer": "활성 레이어의 배경이 제거됨",
        "Content-Aware Fill applied": "콘텐츠 인식 채우기 적용됨",
        "Subject selected": "피사체 선택됨",
        "Upscaled active layer to {w}×{h}": "활성 레이어를 {w}×{h}로 업스케일",
        "Copied merged to clipboard": "병합된 이미지를 클립보드에 복사",
        "Canvas resized {a}×{b} → {c}×{d}": "캔버스 크기 {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "이미지를 {w}×{h}로 변경",
        "Trim: image is empty": "트리밍: 이미지가 비어 있음",
        "Trimmed {a}×{b} → {c}×{d}": "{a}×{b} → {c}×{d}로 잘라냄",
        "Saved {p}": "{p} 저장됨",
    },
    "fr": {
        "Tools": "Outils", "Blend": "Mode de fusion", "Opacity": "Opacité",
        "Visible": "Visible", "Name": "Nom", "Transform": "Transform.",
        "X": "X", "Y": "Y", "Width": "Largeur", "Height": "Hauteur",
        "Rotation": "Rotation", "Flip Horizontal": "Miroir horizontal",
        "Flip Vertical": "Miroir vertical", "origin": "origine",
        "size": "taille", "rotation": "rotation",
        "Mask": "Masque", "Mask enabled": "Masque activé",
        "Invert": "Inverser", "Feather…": "Contour progressif…",
        "Clear": "Effacer", "Apply": "Appliquer", "Edit Text…": "Modifier le texte…",
        "AI Backend Settings": "Paramètres des backends IA",
        "AI background removal applied (U²-Net)": "Suppression d'arrière-plan IA appliquée (U²-Net)",
        "AI Select Subject applied (rembg)": "Sélection du sujet IA appliquée (rembg)",
        "rembg not installed": "rembg n'est pas installé",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "Installer rembg pour la suppression d'arrière-plan IA :\\n\\npip install rembg",
        "Removed background on active layer": "Arrière-plan supprimé sur le calque actif",
        "Content-Aware Fill applied": "Remplissage sensible au contenu appliqué",
        "Subject selected": "Sujet sélectionné",
        "Upscaled active layer to {w}×{h}": "Calque actif agrandi à {w}×{h}",
        "Copied merged to clipboard": "Image fusionnée copiée dans le presse-papiers",
        "Canvas resized {a}×{b} → {c}×{d}": "Canvas redimensionné {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "Image redimensionnée à {w}×{h}",
        "Trim: image is empty": "Recadrage : image vide",
        "Trimmed {a}×{b} → {c}×{d}": "Recadré {a}×{b} → {c}×{d}",
        "Saved {p}": "{p} enregistré",
    },
    "de": {
        "Tools": "Werkzeuge", "Blend": "Mischmodus", "Opacity": "Deckkraft",
        "Visible": "Sichtbar", "Name": "Name", "Transform": "Transformieren",
        "X": "X", "Y": "Y", "Width": "Breite", "Height": "Höhe",
        "Rotation": "Drehung", "Flip Horizontal": "Horizontal spiegeln",
        "Flip Vertical": "Vertikal spiegeln", "origin": "Ursprung",
        "size": "Größe", "rotation": "Drehung",
        "Mask": "Maske", "Mask enabled": "Maske aktiviert",
        "Invert": "Umkehren", "Feather…": "Weichzeichner…",
        "Clear": "Leeren", "Apply": "Anwenden", "Edit Text…": "Text bearbeiten…",
        "AI Backend Settings": "KI-Backend-Einstellungen",
        "AI background removal applied (U²-Net)": "KI-Hintergrundentfernung angewendet (U²-Net)",
        "AI Select Subject applied (rembg)": "KI-Motivauswahl angewendet (rembg)",
        "rembg not installed": "rembg ist nicht installiert",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "rembg für KI-Hintergrundentfernung installieren:\\n\\npip install rembg",
        "Removed background on active layer": "Hintergrund der aktiven Ebene entfernt",
        "Content-Aware Fill applied": "Inhaltsbasierte Füllung angewendet",
        "Subject selected": "Motiv ausgewählt",
        "Upscaled active layer to {w}×{h}": "Aktive Ebene auf {w}×{h} hochskaliert",
        "Copied merged to clipboard": "Zusammengeführtes Bild in die Zwischenablage kopiert",
        "Canvas resized {a}×{b} → {c}×{d}": "Leinwand geändert {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "Bild auf {w}×{h} geändert",
        "Trim: image is empty": "Beschneiden: Bild ist leer",
        "Trimmed {a}×{b} → {c}×{d}": "Beschnitten auf {a}×{b} → {c}×{d}",
        "Saved {p}": "{p} gespeichert",
    },
    "it": {
        "Tools": "Strumenti", "Blend": "Fusione", "Opacity": "Opacità",
        "Visible": "Visibile", "Name": "Nome", "Transform": "Trasforma",
        "X": "X", "Y": "Y", "Width": "Larghezza", "Height": "Altezza",
        "Rotation": "Rotazione", "Flip Horizontal": "Capovolgi orizzontalmente",
        "Flip Vertical": "Capovolgi verticalmente", "origin": "origine",
        "size": "dimensione", "rotation": "rotazione",
        "Mask": "Maschera", "Mask enabled": "Maschera attiva",
        "Invert": "Inverti", "Feather…": "Sfumatura…",
        "Clear": "Cancella", "Apply": "Applica", "Edit Text…": "Modifica testo…",
        "AI Backend Settings": "Impostazioni backend IA",
        "AI background removal applied (U²-Net)": "Rimozione sfondo IA applicata (U²-Net)",
        "AI Select Subject applied (rembg)": "Selezione soggetto IA applicata (rembg)",
        "rembg not installed": "rembg non è installato",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "Installa rembg per la rimozione dello sfondo IA:\\n\\npip install rembg",
        "Removed background on active layer": "Sfondo rimosso sul livello attivo",
        "Content-Aware Fill applied": "Riempimento consapevole del contenuto applicato",
        "Subject selected": "Soggetto selezionato",
        "Upscaled active layer to {w}×{h}": "Livello attivo scalato a {w}×{h}",
        "Copied merged to clipboard": "Immagine unita copiata negli appunti",
        "Canvas resized {a}×{b} → {c}×{d}": "Tela ridimensionata {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "Immagine ridimensionata a {w}×{h}",
        "Trim: image is empty": "Ritaglia: immagine vuota",
        "Trimmed {a}×{b} → {c}×{d}": "Ritagliato {a}×{b} → {c}×{d}",
        "Saved {p}": "{p} salvato",
    },
    "pt": {
        "Tools": "Ferramentas", "Blend": "Mesclagem", "Opacity": "Opacidade",
        "Visible": "Visível", "Name": "Nome", "Transform": "Transformar",
        "X": "X", "Y": "Y", "Width": "Largura", "Height": "Altura",
        "Rotation": "Rotação", "Flip Horizontal": "Inverter horizontal",
        "Flip Vertical": "Inverter vertical", "origin": "origem",
        "size": "tamanho", "rotation": "rotação",
        "Mask": "Máscara", "Mask enabled": "Máscara ativada",
        "Invert": "Inverter", "Feather…": "Suavizar…",
        "Clear": "Limpar", "Apply": "Aplicar", "Edit Text…": "Editar texto…",
        "AI Backend Settings": "Configurações de backend de IA",
        "AI background removal applied (U²-Net)": "Remoção de fundo IA aplicada (U²-Net)",
        "AI Select Subject applied (rembg)": "Seleção de assunto IA aplicada (rembg)",
        "rembg not installed": "rembg não está instalado",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "Instale rembg para remoção de fundo IA:\\n\\npip install rembg",
        "Removed background on active layer": "Fundo removido na camada ativa",
        "Content-Aware Fill applied": "Preenchimento sensível ao conteúdo aplicado",
        "Subject selected": "Assunto selecionado",
        "Upscaled active layer to {w}×{h}": "Camada ativa ampliada para {w}×{h}",
        "Copied merged to clipboard": "Imagem mesclada copiada para a área de transferência",
        "Canvas resized {a}×{b} → {c}×{d}": "Tela redimensionada {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "Imagem redimensionada para {w}×{h}",
        "Trim: image is empty": "Cortar: imagem vazia",
        "Trimmed {a}×{b} → {c}×{d}": "Cortado para {a}×{b} → {c}×{d}",
        "Saved {p}": "{p} salvo",
    },
    "es": {
        "Tools": "Herramientas", "Blend": "Fusión", "Opacity": "Opacidad",
        "Visible": "Visible", "Name": "Nombre", "Transform": "Transformar",
        "X": "X", "Y": "Y", "Width": "Ancho", "Height": "Alto",
        "Rotation": "Rotación", "Flip Horizontal": "Voltear horizontal",
        "Flip Vertical": "Voltear vertical", "origin": "origen",
        "size": "tamaño", "rotation": "rotación",
        "Mask": "Máscara", "Mask enabled": "Máscara activada",
        "Invert": "Invertir", "Feather…": "Desvanecer…",
        "Clear": "Borrar", "Apply": "Aplicar", "Edit Text…": "Editar texto…",
        "AI Backend Settings": "Configuración de backends de IA",
        "AI background removal applied (U²-Net)": "Eliminación de fondo IA aplicada (U²-Net)",
        "AI Select Subject applied (rembg)": "Selección de sujeto IA aplicada (rembg)",
        "rembg not installed": "rembg no está instalado",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "Instale rembg para eliminación de fondo IA:\\n\\npip install rembg",
        "Removed background on active layer": "Fondo eliminado en la capa activa",
        "Content-Aware Fill applied": "Relleno según contenido aplicado",
        "Subject selected": "Sujeto seleccionado",
        "Upscaled active layer to {w}×{h}": "Capa activa escalada a {w}×{h}",
        "Copied merged to clipboard": "Imagen fusionada copiada al portapapeles",
        "Canvas resized {a}×{b} → {c}×{d}": "Lienzo redimensionado {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "Imagen redimensionada a {w}×{h}",
        "Trim: image is empty": "Recortar: imagen vacía",
        "Trimmed {a}×{b} → {c}×{d}": "Recortado a {a}×{b} → {c}×{d}",
        "Saved {p}": "{p} guardado",
    },
    "vi": {
        "Tools": "Công cụ", "Blend": "Hòa trộn", "Opacity": "Độ mờ",
        "Visible": "Hiển thị", "Name": "Tên", "Transform": "Biến đổi",
        "X": "X", "Y": "Y", "Width": "Chiều rộng", "Height": "Chiều cao",
        "Rotation": "Xoay", "Flip Horizontal": "Lật ngang",
        "Flip Vertical": "Lật dọc", "origin": "gốc",
        "size": "kích thước", "rotation": "xoay",
        "Mask": "Mặt nạ", "Mask enabled": "Bật mặt nạ",
        "Invert": "Đảo ngược", "Feather…": "Làm mềm…",
        "Clear": "Xóa", "Apply": "Áp dụng", "Edit Text…": "Sửa văn bản…",
        "AI Backend Settings": "Cài đặt backend AI",
        "AI background removal applied (U²-Net)": "Đã áp dụng tách nền AI (U²-Net)",
        "AI Select Subject applied (rembg)": "Đã áp dụng chọn chủ thể AI (rembg)",
        "rembg not installed": "Chưa cài rembg",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "Cài rembg để tách nền bằng AI:\\n\\npip install rembg",
        "Removed background on active layer": "Đã tách nền trên lớp hiện tại",
        "Content-Aware Fill applied": "Đã áp dụng lấp đầy nhận biết nội dung",
        "Subject selected": "Đã chọn chủ thể",
        "Upscaled active layer to {w}×{h}": "Đã phóng to lớp hiện tại lên {w}×{h}",
        "Copied merged to clipboard": "Đã sao chép ảnh hợp nhất vào bộ nhớ tạm",
        "Canvas resized {a}×{b} → {c}×{d}": "Đã đổi kích thước canvas {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "Đã đổi kích thước ảnh thành {w}×{h}",
        "Trim: image is empty": "Cắt: ảnh trống",
        "Trimmed {a}×{b} → {c}×{d}": "Đã cắt thành {a}×{b} → {c}×{d}",
        "Saved {p}": "Đã lưu {p}",
    },
    "tr": {
        "Tools": "Araçlar", "Blend": "Karıştırma", "Opacity": "Opaklık",
        "Visible": "Görünür", "Name": "Ad", "Transform": "Dönüştür",
        "X": "X", "Y": "Y", "Width": "Genişlik", "Height": "Yükseklik",
        "Rotation": "Döndürme", "Flip Horizontal": "Yatay çevir",
        "Flip Vertical": "Dikey çevir", "origin": "orijin",
        "size": "boyut", "rotation": "döndürme",
        "Mask": "Maske", "Mask enabled": "Maske etkin",
        "Invert": "Ters çevir", "Feather…": "Yumuşat…",
        "Clear": "Temizle", "Apply": "Uygula", "Edit Text…": "Metni düzenle…",
        "AI Backend Settings": "AI arka uç ayarları",
        "AI background removal applied (U²-Net)": "AI arka plan kaldırma uygulandı (U²-Net)",
        "AI Select Subject applied (rembg)": "AI özne seçimi uygulandı (rembg)",
        "rembg not installed": "rembg yüklü değil",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "AI arka plan kaldırma için rembg yükleyin:\\n\\npip install rembg",
        "Removed background on active layer": "Etkin katmanın arka planı kaldırıldı",
        "Content-Aware Fill applied": "İçerik farkında doldurma uygulandı",
        "Subject selected": "Özne seçildi",
        "Upscaled active layer to {w}×{h}": "Etkin katman {w}×{h} olarak büyütüldü",
        "Copied merged to clipboard": "Birleştirilmiş görüntü panoya kopyalandı",
        "Canvas resized {a}×{b} → {c}×{d}": "Tuval boyutu {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "Görüntü {w}×{h} olarak boyutlandırıldı",
        "Trim: image is empty": "Kırp: görüntü boş",
        "Trimmed {a}×{b} → {c}×{d}": "{a}×{b} → {c}×{d} olarak kırpıldı",
        "Saved {p}": "{p} kaydedildi",
    },
    "ar": {
        "Tools": "الأدوات", "Blend": "الدمج", "Opacity": "الشفافية",
        "Visible": "مرئي", "Name": "الاسم", "Transform": "تحويل",
        "X": "س", "Y": "ص", "Width": "العرض", "Height": "الارتفاع",
        "Rotation": "الدوران", "Flip Horizontal": "انعكاس أفقي",
        "Flip Vertical": "انعكاس عمودي", "origin": "الأصل",
        "size": "الحجم", "rotation": "الدوران",
        "Mask": "القناع", "Mask enabled": "القناع مفعّل",
        "Invert": "عكس", "Feather…": "تنعيم…",
        "Clear": "مسح", "Apply": "تطبيق", "Edit Text…": "تحرير النص…",
        "AI Backend Settings": "إعدادات خلفية الذكاء الاصطناعي",
        "AI background removal applied (U²-Net)": "تم تطبيق إزالة الخلفية بالذكاء الاصطناعي (U²-Net)",
        "AI Select Subject applied (rembg)": "تم تطبيق اختيار الموضوع بالذكاء الاصطناعي (rembg)",
        "rembg not installed": "rembg غير مثبت",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "ثبّت rembg لإزالة الخلفية بالذكاء الاصطناعي:\\n\\npip install rembg",
        "Removed background on active layer": "تمت إزالة الخلفية في الطبقة النشطة",
        "Content-Aware Fill applied": "تم تطبيق التعبئة المعتمدة على المحتوى",
        "Subject selected": "تم اختيار الموضوع",
        "Upscaled active layer to {w}×{h}": "تم تكبير الطبقة النشطة إلى {w}×{h}",
        "Copied merged to clipboard": "تم نسخ الصورة المدموجة إلى الحافظة",
        "Canvas resized {a}×{b} → {c}×{d}": "تم تغيير حجم اللوحة {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "تم تغيير حجم الصورة إلى {w}×{h}",
        "Trim: image is empty": "القص: الصورة فارغة",
        "Trimmed {a}×{b} → {c}×{d}": "تم القص إلى {a}×{b} → {c}×{d}",
        "Saved {p}": "تم حفظ {p}",
    },
    "hi": {
        "Tools": "उपकरण", "Blend": "मिश्रण", "Opacity": "अपारदर्शिता",
        "Visible": "दृश्य", "Name": "नाम", "Transform": "रूपांतरण",
        "X": "X", "Y": "Y", "Width": "चौड़ाई", "Height": "ऊँचाई",
        "Rotation": "घूर्णन", "Flip Horizontal": "क्षैतिज पलटें",
        "Flip Vertical": "लंबवत पलटें", "origin": "मूल बिंदु",
        "size": "आकार", "rotation": "घूर्णन",
        "Mask": "मास्क", "Mask enabled": "मास्क सक्षम",
        "Invert": "उलटा", "Feather…": "फेदर…",
        "Clear": "साफ़ करें", "Apply": "लागू करें", "Edit Text…": "पाठ संपादित करें…",
        "AI Backend Settings": "AI बैकएंड सेटिंग्स",
        "AI background removal applied (U²-Net)": "AI पृष्ठभूमि हटाना लागू (U²-Net)",
        "AI Select Subject applied (rembg)": "AI विषय चयन लागू (rembg)",
        "rembg not installed": "rembg स्थापित नहीं है",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "AI पृष्ठभूमि हटाने के लिए rembg स्थापित करें:\\n\\npip install rembg",
        "Removed background on active layer": "सक्रिय परत की पृष्ठभूमि हटाई गई",
        "Content-Aware Fill applied": "सामग्री-आधारित भरण लागू",
        "Subject selected": "विषय चुना गया",
        "Upscaled active layer to {w}×{h}": "सक्रिय परत {w}×{h} तक बड़ी की गई",
        "Copied merged to clipboard": "विलय छवि क्लिपबोर्ड पर कॉपी हुई",
        "Canvas resized {a}×{b} → {c}×{d}": "कैनवास {a}×{b} → {c}×{d} आकार बदला",
        "Image resized to {w}×{h}": "छवि {w}×{h} आकार बदली",
        "Trim: image is empty": "ट्रिम: छवि खाली है",
        "Trimmed {a}×{b} → {c}×{d}": "{a}×{b} → {c}×{d} तक ट्रिम किया",
        "Saved {p}": "{p} सहेजा गया",
    },
    "id": {
        "Tools": "Peralatan", "Blend": "Campur", "Opacity": "Opasitas",
        "Visible": "Terlihat", "Name": "Nama", "Transform": "Ubah",
        "X": "X", "Y": "Y", "Width": "Lebar", "Height": "Tinggi",
        "Rotation": "Rotasi", "Flip Horizontal": "Balik horizontal",
        "Flip Vertical": "Balik vertikal", "origin": "asal",
        "size": "ukuran", "rotation": "rotasi",
        "Mask": "Masker", "Mask enabled": "Masker aktif",
        "Invert": "Balik", "Feather…": "Bulu…",
        "Clear": "Bersihkan", "Apply": "Terapkan", "Edit Text…": "Sunting teks…",
        "AI Backend Settings": "Pengaturan backend AI",
        "AI background removal applied (U²-Net)": "Penghapusan latar AI diterapkan (U²-Net)",
        "AI Select Subject applied (rembg)": "Pemilihan subjek AI diterapkan (rembg)",
        "rembg not installed": "rembg belum terpasang",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "Pasang rembg untuk penghapusan latar AI:\\n\\npip install rembg",
        "Removed background on active layer": "Latar dihapus pada layer aktif",
        "Content-Aware Fill applied": "Pengisian berbasis konten diterapkan",
        "Subject selected": "Subjek dipilih",
        "Upscaled active layer to {w}×{h}": "Layer aktif diperbesar ke {w}×{h}",
        "Copied merged to clipboard": "Gambar gabungan disalin ke papan klip",
        "Canvas resized {a}×{b} → {c}×{d}": "Kanvas diubah {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "Gambar diubah menjadi {w}×{h}",
        "Trim: image is empty": "Pangkas: gambar kosong",
        "Trimmed {a}×{b} → {c}×{d}": "Dipangkas menjadi {a}×{b} → {c}×{d}",
        "Saved {p}": "{p} tersimpan",
    },
    "th": {
        "Tools": "เครื่องมือ", "Blend": "ผสม", "Opacity": "ความทึบ",
        "Visible": "มองเห็น", "Name": "ชื่อ", "Transform": "แปลง",
        "X": "X", "Y": "Y", "Width": "กว้าง", "Height": "สูง",
        "Rotation": "หมุน", "Flip Horizontal": "พลิกแนวนอน",
        "Flip Vertical": "พลิกแนวตั้ง", "origin": "จุดกำเนิด",
        "size": "ขนาด", "rotation": "การหมุน",
        "Mask": "มาสก์", "Mask enabled": "เปิดใช้มาสก์",
        "Invert": "กลับสี", "Feather…": "เฟเธอร์…",
        "Clear": "ล้าง", "Apply": "ใช้", "Edit Text…": "แก้ไขข้อความ…",
        "AI Backend Settings": "การตั้งค่าแบ็กเอนด์ AI",
        "AI background removal applied (U²-Net)": "ลบพื้นหลังด้วย AI แล้ว (U²-Net)",
        "AI Select Subject applied (rembg)": "เลือกวัตถุด้วย AI แล้ว (rembg)",
        "rembg not installed": "ยังไม่ได้ติดตั้ง rembg",
        "Install rembg for AI-powered background removal:\n\npip install rembg":
            "ติดตั้ง rembg เพื่อลบพื้นหลังด้วย AI:\\n\\npip install rembg",
        "Removed background on active layer": "ลบพื้นหลังของเลเยอร์ที่ใช้งานอยู่",
        "Content-Aware Fill applied": "ใช้การเติมตามเนื้อหาแล้ว",
        "Subject selected": "เลือกวัตถุแล้ว",
        "Upscaled active layer to {w}×{h}": "ขยายเลเยอร์ที่ใช้งานอยู่เป็น {w}×{h}",
        "Copied merged to clipboard": "คัดลอกภาพที่ผสานแล้วไปยังคลิปบอร์ด",
        "Canvas resized {a}×{b} → {c}×{d}": "ปรับขนาดผืนผ้าใบ {a}×{b} → {c}×{d}",
        "Image resized to {w}×{h}": "ปรับขนาดภาพเป็น {w}×{h}",
        "Trim: image is empty": "ครอบตัด: ภาพว่าง",
        "Trimmed {a}×{b} → {c}×{d}": "ครอบตัดเป็น {a}×{b} → {c}×{d}",
        "Saved {p}": "บันทึก {p} แล้ว",
    },
}


def main() -> int:
    with open(JSON_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    n_added = 0
    for code, overrides in TRANSLATIONS.items():
        if code not in data:
            print(f"[warn] {code} not in i18n_data.json")
            continue
        for key, value in overrides.items():
            if key not in data["en"]:
                print(f"[warn] key {key!r} not in EN")
                continue
            data[code][key] = value
            n_added += 1

    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"Applied {n_added} translations across {len(TRANSLATIONS)} languages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
