# Simple test - save as test_simple.py in your ROOT directory
import asyncio
from datetime import datetime

# Direct import (adjust path as needed)
from services.database.ai_system import AdvancedCryptoAI

async def quick_test():
    print("🧪 Quick AI Test...")
    
    # Mock data
    class Purchase:
        def __init__(self, token, eth):
            self.token_bought = token
            self.eth_spent = eth
            self.amount_received = 1000000
            self.wallet_address = "0xtest"
            self.sophistication_score = 250
            self.timestamp = datetime.now()
            self.transaction_hash = "0xtest"
            self.web3_analysis = {'contract_address': "0xcontract"}
    
    purchases = [Purchase("TEST", 2.0), Purchase("TEST", 1.5)]
    
    # Test AI
    ai = AdvancedCryptoAI()
    result = await ai.complete_ai_analysis(purchases, "buy")
    
    if result.get('enhanced'):
        print("✅ AI working!")
        scores = result.get('scores', {})
        if scores:
            token, data = next(iter(scores.items()))
            print(f"🏆 {token}: {data['total_score']:.1f} (confidence: {data['confidence']:.1%})")
    else:
        print("❌ AI failed:", result.get('error'))

if __name__ == "__main__":
    asyncio.run(quick_test())