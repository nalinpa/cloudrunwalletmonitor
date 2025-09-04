#!/usr/bin/env python3
"""
Telegram Notification Debug Script
Test your Telegram bot configuration and send test messages
"""
import os
import asyncio
import httpx
import json
from datetime import datetime

class TelegramDebugger:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        
    def check_environment(self):
        """Check if environment variables are set correctly"""
        print("🔍 Checking Environment Variables...")
        
        if not self.bot_token:
            print("❌ TELEGRAM_BOT_TOKEN not found")
            return False
        
        if not self.chat_id:
            print("❌ TELEGRAM_CHAT_ID not found")
            return False
            
        print(f"✅ TELEGRAM_BOT_TOKEN: {self.bot_token[:10]}...{self.bot_token[-5:]}")
        print(f"✅ TELEGRAM_CHAT_ID: {self.chat_id}")
        
        # Validate token format
        if len(self.bot_token) < 40 or ':' not in self.bot_token:
            print("❌ Bot token format looks invalid")
            return False
            
        # Validate chat ID format
        try:
            int(self.chat_id)
        except ValueError:
            print("❌ Chat ID should be numeric")
            return False
            
        print("✅ Format validation passed")
        return True
    
    async def test_bot_info(self):
        """Test if bot token is valid"""
        if not self.base_url:
            return False
            
        print("\n🤖 Testing Bot Token...")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/getMe")
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        bot_info = data.get('result', {})
                        print(f"✅ Bot is valid: @{bot_info.get('username', 'unknown')}")
                        print(f"   Bot name: {bot_info.get('first_name', 'Unknown')}")
                        print(f"   Bot ID: {bot_info.get('id', 'Unknown')}")
                        return True
                    else:
                        print(f"❌ Bot API error: {data.get('description', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ HTTP {response.status_code}: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Bot test failed: {e}")
            return False
    
    async def test_chat_access(self):
        """Test if bot can access the chat"""
        if not self.base_url:
            return False
            
        print("\n💬 Testing Chat Access...")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try to get chat info
                response = await client.get(
                    f"{self.base_url}/getChat",
                    params={"chat_id": self.chat_id}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        chat_info = data.get('result', {})
                        chat_type = chat_info.get('type', 'unknown')
                        
                        if chat_type == 'private':
                            print(f"✅ Private chat with: {chat_info.get('first_name', 'Unknown')}")
                        elif chat_type in ['group', 'supergroup']:
                            print(f"✅ Group chat: {chat_info.get('title', 'Unknown Group')}")
                        else:
                            print(f"✅ Chat type: {chat_type}")
                            
                        return True
                    else:
                        error_desc = data.get('description', 'Unknown error')
                        print(f"❌ Chat access error: {error_desc}")
                        
                        if 'bot was blocked' in error_desc.lower():
                            print("💡 Solution: Unblock the bot and send /start")
                        elif 'chat not found' in error_desc.lower():
                            print("💡 Solution: Check your chat ID is correct")
                        elif 'forbidden' in error_desc.lower():
                            print("💡 Solution: Add bot to group or start private chat")
                            
                        return False
                else:
                    print(f"❌ HTTP {response.status_code}: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Chat access test failed: {e}")
            return False
    
    async def send_test_message(self):
        """Send a test message"""
        if not self.base_url:
            return False
            
        print("\n📨 Sending Test Message...")
        
        test_message = f"""🧪 **TELEGRAM TEST MESSAGE**

✅ **Bot Configuration:** Working
🕐 **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🚀 **Source:** Debug Script
🌐 **Status:** Connection Successful

If you see this message, your Telegram notifications are working! 🎉"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "chat_id": self.chat_id,
                    "text": test_message,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                }
                
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        message_info = data.get('result', {})
                        print(f"✅ Message sent successfully!")
                        print(f"   Message ID: {message_info.get('message_id', 'Unknown')}")
                        return True
                    else:
                        print(f"❌ Message send error: {data.get('description', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ HTTP {response.status_code}: {response.text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Message send failed: {e}")
            return False
    
    async def test_markdown_formatting(self):
        """Test markdown formatting"""
        print("\n🎨 Testing Markdown Formatting...")
        
        markdown_test = """🎨 **Markdown Test**

**Bold Text**
*Italic Text*
`Code Text`
[Link](https://telegram.org)

✅ If formatting works, notifications will look good!"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "chat_id": self.chat_id,
                    "text": markdown_test,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True
                }
                
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok'):
                        print("✅ Markdown formatting test successful!")
                        return True
                    else:
                        print(f"❌ Markdown test error: {data.get('description', 'Unknown')}")
                        return False
                else:
                    return False
                    
        except Exception as e:
            print(f"❌ Markdown test failed: {e}")
            return False
    
    async def run_full_test(self):
        """Run complete test suite"""
        print("🚀 Starting Telegram Debug Test Suite")
        print("=" * 50)
        
        # Step 1: Environment check
        if not self.check_environment():
            print("\n❌ Environment check failed. Fix your .env configuration first.")
            return False
        
        # Step 2: Bot token test
        if not await self.test_bot_info():
            print("\n❌ Bot token test failed. Check your TELEGRAM_BOT_TOKEN.")
            return False
        
        # Step 3: Chat access test
        if not await self.test_chat_access():
            print("\n❌ Chat access test failed. Check your TELEGRAM_CHAT_ID and bot permissions.")
            return False
        
        # Step 4: Send test message
        if not await self.send_test_message():
            print("\n❌ Test message failed.")
            return False
        
        # Step 5: Test markdown
        await asyncio.sleep(1)  # Rate limit
        if not await self.test_markdown_formatting():
            print("\n❌ Markdown test failed.")
            return False
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your Telegram bot is fully configured and working!")
        print("🚀 Notifications should work in your crypto analysis system.")
        
        return True

async def main():
    """Main test function"""
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv('.env.production')
        print("📁 Loaded .env.production")
    except ImportError:
        print("⚠️ python-dotenv not installed, using system environment")
    except FileNotFoundError:
        print("⚠️ .env.production not found, using system environment")
    
    debugger = TelegramDebugger()
    success = await debugger.run_full_test()
    
    if not success:
        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("1. Make sure your bot token is correct")
        print("2. Ensure the bot can access your chat (start private chat or add to group)")
        print("3. Check that TELEGRAM_CHAT_ID is your user ID or group chat ID")
        print("4. For groups, make sure the bot is added as member")
        print("5. Try sending /start to your bot in private message")

if __name__ == "__main__":
    asyncio.run(main())