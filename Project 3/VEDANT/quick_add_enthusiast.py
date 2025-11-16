import weaviate
import requests
import time
from sentence_transformers import SentenceTransformer

def quick_add_enthusiast_cars():
    print("🏎️  QUICK ADD: Enthusiast Cars")
    
    # Wait for Weaviate
    for i in range(10):
        try:
            response = requests.get("http://localhost:8080/v1/meta", timeout=5)
            if response.status_code == 200:
                print("✅ Weaviate is ready!")
                break
        except:
            pass
        time.sleep(1)
    
    client = weaviate.Client("http://localhost:8080")
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Focus on key enthusiast cars you asked about
    enthusiast_cars = [
        {
            "manufacturer": "Force", "model": "Gurkha", "variant": "3-door", "year": 2024,
            "price_min": 14.0, "price_max": 16.0, "fuel_type": "Diesel", "body_type": "SUV",
            "mileage": "8-10 kmpl", "engine": "2.6L Diesel", "transmission": "Manual",
            "safety_rating": "3", "features": "4WD, front and rear differential locks, snorkel, off-road tires",
            "pros": "Extreme off-road capability, utilitarian design, expedition-ready",
            "cons": "Uncomfortable for daily use, basic interiors, poor highway manners",
            "description": "Force Gurkha 3-door is a hardcore off-roader that rivals the Thar. Enthusiasts love its utilitarian design and expedition-ready feel with proper 4x4 hardware. It's a no-nonsense off-roader built for serious adventure.",
            "source_url": ""
        },
        {
            "manufacturer": "Force", "model": "Gurkha", "variant": "5-door", "year": 2024,
            "price_min": 16.0, "price_max": 18.0, "fuel_type": "Diesel", "body_type": "SUV",
            "mileage": "9-11 kmpl", "engine": "2.6L Diesel", "transmission": "Manual", 
            "safety_rating": "3", "features": "4WD, diff locks, snorkel, off-road tires, slightly more space",
            "pros": "Better practicality than 3-door, same off-road capability, more family-friendly",
            "cons": "Still basic, not for city use, limited features",
            "description": "Gurkha 5-door offers the same extreme off-road capability with slightly more practicality. Loved by adventure seekers who need some extra space without compromising on off-road prowess.",
            "source_url": ""
        },
        {
            "manufacturer": "Toyota", "model": "Fortuner", "variant": "Legender", "year": 2024,
            "price_min": 32.0, "price_max": 38.0, "fuel_type": "Diesel", "body_type": "SUV",
            "mileage": "10-12 kmpl", "engine": "2.8L Turbo Diesel", "transmission": "Automatic",
            "safety_rating": "5", "features": "7-seater, LED DRLs, 8-inch touchscreen, 6 airbags, premium audio",
            "pros": "Powerful engine, premium cabin, excellent off-road capability, Toyota reliability",
            "cons": "Low fuel efficiency, expensive maintenance, large size",
            "description": "Toyota Fortuner Legender is the premium variant with distinctive styling and robust performance. Loved by enthusiasts for its commanding road presence and bulletproof reliability. It dominates Indian roads with its sheer presence.",
            "source_url": ""
        },
        {
            "manufacturer": "Skoda", "model": "Octavia", "variant": "vRS", "year": 2023,
            "price_min": 45.0, "price_max": 55.0, "fuel_type": "Petrol", "body_type": "Sedan", 
            "mileage": "12-14 kmpl", "engine": "2.0L TSI", "transmission": "Automatic",
            "safety_rating": "5", "features": "Sport suspension, bucket seats, digital cockpit, performance brakes",
            "pros": "Turbocharged performance, European driving dynamics, cult following, insane remap potential",
            "cons": "Expensive, limited availability, high maintenance costs",
            "description": "Skoda Octavia vRS is a turbocharged performance sedan with cult following among enthusiasts. Loved for its insane remap potential and sharp European driving dynamics. It's a proper hot sedan that delivers thrilling performance.",
            "source_url": ""
        }
    ]
    
    print(f"📊 Adding {len(enthusiast_cars)} key enthusiast cars...")
    
    for car in enthusiast_cars:
        text_content = (
            f"{car['manufacturer']} {car['model']} {car['variant']} ({car['year']}). "
            f"Price: ₹{car['price_min']}-{car['price_max']} lakhs. "
            f"{car['description']} "
            f"Key features: {car['features']}. "
            f"Pros: {car['pros']}. Cons: {car['cons']}."
        )
        
        vector = embedder.encode(text_content).tolist()
        
        obj = {
            "manufacturer": car['manufacturer'],
            "model": car['model'], 
            "variant": car['variant'],
            "year": car['year'],
            "price_min": car['price_min'],
            "price_max": car['price_max'],
            "fuel_type": car['fuel_type'],
            "body_type": car['body_type'],
            "mileage": car['mileage'],
            "features": car['features'],
            "pros": car['pros'],
            "cons": car['cons'],
            "text_content": text_content,
            "source_url": car['source_url']
        }
        
        try:
            client.data_object.create(
                data_object=obj,
                class_name="IndianCar", 
                vector=vector
            )
            print(f"✅ Added {car['manufacturer']} {car['model']} {car['variant']}")
        except Exception as e:
            print(f"❌ Failed to add {car['manufacturer']} {car['model']}: {e}")
    
    print("🎉 Enthusiast cars added!")

if __name__ == "__main__":
    quick_add_enthusiast_cars()