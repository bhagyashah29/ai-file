from __future__ import annotations


Intent = str


def classify_intent(text: str) -> Intent:
    lowered = text.lower()
    if lowered.strip().startswith("#"):
        return "staff_command"
    if any(word in lowered for word in ["complaint", "dirty", "late", "refund", "problem", "issue", "ganda", "pareshan"]):
        return "complaint"
    if any(word in lowered for word in ["book", "booking", "reserve", "available", "room", "tariff", "price", "kitna", "rate"]):
        return "booking_inquiry"
    if any(word in lowered for word in ["payment", "pay", "upi", "link", "discount", "coupon", "promo"]):
        return "booking_inquiry"
    if any(word in lowered for word in ["wifi", "pool", "gym", "breakfast", "pickup", "parking", "amenity", "suvidha"]):
        return "amenity_question"
    return "other"
