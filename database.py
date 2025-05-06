from pymongo import MongoClient
import datetime

client = MongoClient("mongodb://localhost:27017/")
db = client["SmartShopDB"]
collection = db["customers"]

def get_customer_by_name(name: str):
    return collection.find_one({"name": name})

def add_customer(data: dict):
    collection.insert_one(data)

def update_customer(name: str, update: dict):
    collection.update_one({"name": name}, {"$set": update})
