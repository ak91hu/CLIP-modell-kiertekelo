import glob
import os
from PIL import Image
from transformers import pipeline

print("Modell betöltése... (ez eltarthat egy ideig)")
classifier = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)
print("Modell betöltve.")

candidate_labels = [
    "speed limit", 
    "bicycles only", 
    "zebra crossing", 
    "no stopping", 
    "no entry restriction"
]

prefix_to_label = {
    '003': 'speed limit',
    '005': 'speed limit',
    '007': 'speed limit',
    '030': 'bicycles only',
    '035': 'zebra crossing',
    '054': 'no stopping',
    '055': 'no entry restriction'
}

uncertain_counts = {label: 0 for label in candidate_labels}
misclassified_counts = {label: 0 for label in candidate_labels}
total_counts = {label: 0 for label in candidate_labels}

image_files = glob.glob("data/*") 
print(f"Összesen {len(image_files)} kép feldolgozása...")

for image_path in image_files:
    try:
        image = Image.open(image_path)
        filename = os.path.basename(image_path)
        prefix = filename[:3]
        true_label = prefix_to_label.get(prefix)
        
        if not true_label:
            continue
            
        result = classifier(image, candidate_labels=candidate_labels)
        
        top_prediction = result[0]
        top_label = top_prediction['label']
        top_score = top_prediction['score']
        
        total_counts[true_label] += 1
        
        if top_score <= 0.50:
            uncertain_counts[true_label] += 1
        else:
            if top_label != true_label:
                misclassified_counts[true_label] += 1
            
    except Exception as e:
        print(f"Hiba a(z) {image_path} feldolgozása közben: {e}")

print("\n" + "="*30)
print("TELJES EREDMÉNY KIMUTATÁS")
print("="*30)

print("\n--- Eredmény (1. kérdés: BIZONYTALAN) ---")
print("Azon típusok, ahol a modell 50% alatti valószínűséget adott:\n")

most_uncertain = [
    (label, count) for label, count in uncertain_counts.items()
    if count > 0 and count == max(uncertain_counts.values())
]
for label in candidate_labels:
    total = total_counts.get(label, 0)
    uncertain = uncertain_counts.get(label, 0)
    if total > 0:
        highlight = " <<< LEGTÖBB" if (label, uncertain) in most_uncertain else ""
        print(f"- {label:<20}: {uncertain:>3} / {total:<3} alkalommal volt bizonytalan {highlight}")
print("\n >> VÁLASZ (1. kérdés):", " ".join([label for label, count in most_uncertain]))

print("\n--- Eredmény (2. kérdés: TÉVES) ---")
print("Azon típusok, ahol a modell >50% valószínűséggel, de ROSSZ címkét adott:\n")

most_misclassified = [
    (label, count) for label, count in misclassified_counts.items()
    if count > 0 and count == max(misclassified_counts.values())
]
for label in candidate_labels:
    total = total_counts.get(label, 0)
    misclassified = misclassified_counts.get(label, 0)
    if total > 0:
        highlight = " <<< LEGTÖBB" if (label, misclassified) in most_misclassified else ""
        print(f"- {label:<20}: {misclassified:>3} / {total:<3} alkalommal volt téves {highlight}")
print("\n >> VÁLASZ (2. kérdés):", " ".join([label for label, count in most_misclassified]))

print("\n--- Eredmény (3. kérdés: MINDIG HELYES) ---")
print("Azon típusok, ahol a modell 0-szor volt bizonytalan ÉS 0-szor tévedett:\n")

always_correct = []
for label in candidate_labels:
    total = total_counts.get(label, 0)
    uncertain = uncertain_counts.get(label, 0)
    misclassified = misclassified_counts.get(label, 0)
    
    if total > 0:
        if uncertain == 0 and misclassified == 0:
            always_correct.append(label)
            print(f"- {label:<20}: {total:>3} / {total:<3} alkalommal helyes (100%) <<< VÁLASZ")
        else:
            correct_count = total - uncertain - misclassified
            print(f"- {label:<20}: {correct_count:>3} / {total:<3} alkalommal helyes")

print("\n >> VÁLASZ (3. kérdés):", " ".join(always_correct))
print("\n" + "="*30)
