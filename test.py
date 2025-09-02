import asyncio
import os
from datetime import datetime, UTC
from dotenv import load_dotenv

async def simple_transfer_test():
    """Simple test to store transfers with the fixed service"""
    
    print("🧪 SIMPLE TRANSFER STORAGE TEST")
    print("=" * 40)
    
    # Load environment
    load_dotenv('.env.production')
    
    if not os.getenv('MONGO_URI'):
        print("❌ MONGO_URI not loaded!")
        return
    
    print("✅ Environment loaded")
    
    try:
        # Replace the transfer service file first
        print("⚠️ Make sure you've updated services/database/transfer_service.py with the fixed version!")
        input("Press Enter to continue after updating the file...")
        
        from utils.config import Config
        from services.database.transfer_service import TransferService
        from api.models.data_models import Transfer, TransferType
        
        print("✅ Imports successful")
        
        # Initialize
        config = Config()
        transfer_service = TransferService(config)
        await transfer_service.initialize()
        
        if not transfer_service._connected:
            print("❌ Transfer service not connected")
            return
        
        print("✅ Transfer service connected")
        
        # Create test transfers
        print("1️⃣ Creating test transfers...")
        
        test_transfers = []
        for i in range(3):  # Just 3 test transfers
            transfer = Transfer(
                wallet_address=f"0x{'1234567890abcdef' * 2}{i:08x}",
                token_address=f"0x{'abcdef1234567890' * 2}{i:08x}",
                transfer_type=TransferType.BUY if i % 2 == 0 else TransferType.SELL,
                timestamp=datetime.now(UTC),
                cost_in_eth=0.1 * (i + 1),
                transaction_hash=f"0x{'fedcba0987654321' * 3}{i:08x}",
                block_number=20000000 + i,
                token_amount=1000.0 * (i + 1),
                token_symbol=f"TEST{i+1}",
                network="ethereum",
                platform="TestDEX",
                wallet_sophistication_score=50.0 + i * 10
            )
            test_transfers.append(transfer)
        
        print(f"✅ Created {len(test_transfers)} test transfers")
        
        # Test Method 1: Batch storage
        print("2️⃣ Testing batch storage...")
        stored_count = await transfer_service.store_transfers_batch(test_transfers)
        print(f"✅ Batch storage result: {stored_count} transfers stored")
        
        # Check if collection was created
        collections = await transfer_service.db.list_collection_names()
        print(f"📋 Collections: {collections}")
        
        if 'transfers' in collections:
            count = await transfer_service.collection.count_documents({})
            print(f"✅ Transfers collection has {count} documents")
            
            # Show samples
            if count > 0:
                print("📄 Sample transfers:")
                async for doc in transfer_service.collection.find().limit(3):
                    print(f"  - {doc.get('token_symbol')}: {doc.get('transfer_type')} | {doc.get('cost_in_eth')} ETH")
                
                # Test Method 2: Simple storage if batch failed
                if stored_count == 0:
                    print("\n3️⃣ Batch failed, trying simple storage...")
                    simple_stored = await transfer_service.store_transfers_simple([test_transfers[0]])
                    print(f"✅ Simple storage result: {simple_stored} transfers stored")
                    
                    if simple_stored > 0:
                        final_count = await transfer_service.collection.count_documents({})
                        print(f"✅ Final count: {final_count} documents")
                
                # Clean up test data
                print("\n4️⃣ Cleaning up test data...")
                for transfer in test_transfers:
                    await transfer_service.collection.delete_one({
                        "transaction_hash": transfer.transaction_hash
                    })
                print("🗑️ Test data cleaned up")
            
        else:
            print("❌ Transfers collection was not created")
            
            # Try simple storage method
            print("\n3️⃣ Trying simple storage method...")
            simple_stored = await transfer_service.store_transfers_simple([test_transfers[0]])
            print(f"✅ Simple storage result: {simple_stored}")
            
            if simple_stored > 0:
                print("✅ Simple storage worked! Collection should now exist")
                collections = await transfer_service.db.list_collection_names()
                print(f"📋 Collections now: {collections}")
        
        await transfer_service.cleanup()
        print("✅ Test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_transfer_test())