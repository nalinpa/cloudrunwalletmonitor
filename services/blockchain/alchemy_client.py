import aiohttp
import asyncio
import logging
from typing import List, Dict, Tuple
from utils.config import Config

logger = logging.getLogger(__name__)

class AlchemyService:
    """Alchemy API client for blockchain data"""
    
    def __init__(self, config: Config):
        self.config = config
    
    async def get_block_range(self, network: str, days_back: float) -> Tuple[int, int]:
        """Get block range for the specified time period"""
        base_url = self.config.alchemy_endpoints.get(network)
        if not base_url:
            return 0, 0
        
        blocks_per_hour = self.config.blocks_per_hour.get(network, 300)
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get latest block
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_blockNumber",
                    "params": []
                }
                
                async with session.post(base_url, json=payload) as response:
                    data = await response.json()
                    end_block = int(data['result'], 16)
                
                # Calculate start block
                blocks_back = int(days_back * 24 * blocks_per_hour)
                start_block = max(0, end_block - blocks_back)
                
                return start_block, end_block
                
        except Exception as e:
            logger.error(f"Error getting block range: {e}")
            return 0, 0
    
    async def get_transfers_batch(self, network: str, wallet_addresses: List[str], 
                                start_block: int, end_block: int) -> Dict:
        """Get transfers for multiple wallets"""
        base_url = self.config.alchemy_endpoints.get(network)
        if not base_url:
            return {}
        
        all_transfers = {}
        batch_size = 10  # Smaller batches for cloud functions
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            for i in range(0, len(wallet_addresses), batch_size):
                batch_addresses = wallet_addresses[i:i + batch_size]
                
                # Process each address in the batch
                tasks = []
                for address in batch_addresses:
                    task = self._get_address_transfers(session, base_url, address, start_block, end_block)
                    tasks.append(task)
                
                # Execute batch concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for address, result in zip(batch_addresses, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching transfers for {address}: {result}")
                        all_transfers[address] = {"incoming": [], "outgoing": []}
                    else:
                        all_transfers[address] = result
                
                # Rate limiting
                await asyncio.sleep(0.1)
        
        return all_transfers
    
    async def _get_address_transfers(self, session: aiohttp.ClientSession, 
                                   base_url: str, address: str, 
                                   start_block: int, end_block: int) -> Dict:
        """Get transfers for a single address"""
        try:
            # Get incoming transfers
            incoming_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "alchemy_getAssetTransfers",
                "params": [{
                    "fromBlock": hex(start_block),
                    "toBlock": hex(end_block),
                    "toAddress": address,
                    "category": ["erc20"],
                    "withMetadata": False,
                    "excludeZeroValue": True,
                    "maxCount": "0x64"  # Limit to 100 transfers
                }]
            }
            
            async with session.post(base_url, json=incoming_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    incoming = data.get('result', {}).get('transfers', [])
                else:
                    incoming = []
            
            # Get outgoing transfers
            outgoing_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "alchemy_getAssetTransfers",
                "params": [{
                    "fromBlock": hex(start_block),
                    "toBlock": hex(end_block),
                    "fromAddress": address,
                    "category": ["erc20"],
                    "withMetadata": False,
                    "excludeZeroValue": True,
                    "maxCount": "0x64"
                }]
            }
            
            async with session.post(base_url, json=outgoing_payload) as response:
                if response.status == 200:
                    data = await response.json()
                    outgoing = data.get('result', {}).get('transfers', [])
                else:
                    outgoing = []
            
            return {
                "incoming": incoming,
                "outgoing": outgoing
            }
            
        except Exception as e:
            logger.error(f"Error fetching transfers for {address}: {e}")
            return {"incoming": [], "outgoing": []}