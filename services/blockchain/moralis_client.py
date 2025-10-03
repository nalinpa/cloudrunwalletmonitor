import aiohttp
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from utils.config import Config

logger = logging.getLogger(__name__)

class MoralisService:
    """Moralis API client for blockchain data - replaces Alchemy"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = "https://deep-index.moralis.io/api/v2.2"
        self.api_key = config.moralis_api_key
        
        # Rate limiting
        self.rate_limit_delay = 0.2  # 5 requests per second for free tier
        self.last_request_time = 0
        
    async def _rate_limit(self):
        """Apply rate limiting between requests"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    def _get_chain_id(self, network: str) -> str:
        """Convert network name to Moralis chain identifier"""
        chain_map = {
            'ethereum': '0x1',
            'base': '0x2105'  # Base mainnet
        }
        return chain_map.get(network.lower(), '0x1')
    
    async def get_block_range(self, network: str, days_back: float) -> Tuple[int, int]:
        """Get block range for the specified time period using Moralis"""
        try:
            chain = self._get_chain_id(network)
            
            async with aiohttp.ClientSession() as session:
                # Get current block number
                url = f"{self.base_url}/block/date"
                
                # Calculate target date
                target_date = datetime.utcnow() - timedelta(days=days_back)
                
                headers = {
                    "accept": "application/json",
                    "X-API-Key": self.api_key
                }
                
                params = {
                    "chain": chain,
                    "date": target_date.strftime("%Y-%m-%d")
                }
                
                await self._rate_limit()
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        start_block = data.get('block', 0)
                        
                        # Get current block
                        params['date'] = datetime.utcnow().strftime("%Y-%m-%d")
                        
                        await self._rate_limit()
                        
                        async with session.get(url, headers=headers, params=params) as curr_response:
                            if curr_response.status == 200:
                                curr_data = await curr_response.json()
                                end_block = curr_data.get('block', 0)
                                
                                logger.info(f"Moralis block range: {start_block} to {end_block}")
                                return start_block, end_block
            
            # Fallback to approximate calculation
            blocks_per_hour = self.config.blocks_per_hour.get(network, 300)
            blocks_back = int(days_back * 24 * blocks_per_hour)
            
            # Get latest block via eth_blockNumber equivalent
            end_block = await self._get_latest_block(network)
            start_block = max(0, end_block - blocks_back)
            
            return start_block, end_block
                
        except Exception as e:
            logger.error(f"Error getting block range from Moralis: {e}")
            return 0, 0
    
    async def _get_latest_block(self, network: str) -> int:
        """Get latest block number"""
        try:
            chain = self._get_chain_id(network)
            
            async with aiohttp.ClientSession() as session:
                # Use date endpoint with current date
                url = f"{self.base_url}/block/date"
                
                headers = {
                    "accept": "application/json",
                    "X-API-Key": self.api_key
                }
                
                params = {
                    "chain": chain,
                    "date": datetime.utcnow().strftime("%Y-%m-%d")
                }
                
                await self._rate_limit()
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('block', 0)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error getting latest block: {e}")
            return 0
    
    async def get_transfers_batch(self, network: str, wallet_addresses: List[str], 
                                start_block: int, end_block: int) -> Dict:
        """Get ERC20 transfers for multiple wallets using Moralis"""
        chain = self._get_chain_id(network)
        all_transfers = {}
        
        # Process wallets in smaller batches
        batch_size = 5  # Conservative for rate limiting
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            for i in range(0, len(wallet_addresses), batch_size):
                batch_addresses = wallet_addresses[i:i + batch_size]
                
                # Process each address
                tasks = []
                for address in batch_addresses:
                    task = self._get_address_erc20_transfers(
                        session, chain, address, start_block, end_block
                    )
                    tasks.append(task)
                
                # Execute batch
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for address, result in zip(batch_addresses, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching transfers for {address}: {result}")
                        all_transfers[address] = {"incoming": [], "outgoing": []}
                    else:
                        all_transfers[address] = result
                
                # Rate limiting between batches
                if i + batch_size < len(wallet_addresses):
                    await asyncio.sleep(1)  # Extra delay between batches
        
        return all_transfers
    
    async def _get_address_erc20_transfers(self, session: aiohttp.ClientSession,
                                          chain: str, address: str,
                                          start_block: int, end_block: int) -> Dict:
        """Get ERC20 transfers for a single address using Moralis"""
        try:
            url = f"{self.base_url}/{address}/erc20/transfers"
            
            headers = {
                "accept": "application/json",
                "X-API-Key": self.api_key
            }
            
            params = {
                "chain": chain,
                "from_block": start_block,
                "to_block": end_block,
                "limit": 100  # Maximum per request
            }
            
            await self._rate_limit()
            
            all_transfers = []
            cursor = None
            
            # Handle pagination
            while True:
                if cursor:
                    params['cursor'] = cursor
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        transfers = data.get('result', [])
                        all_transfers.extend(transfers)
                        
                        # Check for more pages
                        cursor = data.get('cursor')
                        if not cursor or len(transfers) == 0:
                            break
                        
                        # Rate limit for pagination
                        await self._rate_limit()
                    else:
                        logger.warning(f"Moralis API error {response.status} for {address}")
                        break
            
            # Separate incoming and outgoing
            incoming = []
            outgoing = []
            
            for transfer in all_transfers:
                # Convert Moralis format to Alchemy-like format for compatibility
                converted = self._convert_moralis_to_alchemy_format(transfer, address)
                
                if transfer['to_address'].lower() == address.lower():
                    incoming.append(converted)
                elif transfer['from_address'].lower() == address.lower():
                    outgoing.append(converted)
            
            logger.info(f"Moralis: {address[:10]}... - {len(incoming)} in, {len(outgoing)} out")
            
            return {
                "incoming": incoming,
                "outgoing": outgoing
            }
            
        except Exception as e:
            logger.error(f"Error fetching Moralis transfers for {address}: {e}")
            return {"incoming": [], "outgoing": []}
    
    def _convert_moralis_to_alchemy_format(self, moralis_transfer: Dict, wallet_address: str) -> Dict:
        """Convert Moralis transfer format to Alchemy-compatible format"""
        try:
            # Moralis format:
            # {
            #   "transaction_hash": "0x...",
            #   "address": "0x..." (token contract),
            #   "from_address": "0x...",
            #   "to_address": "0x...",
            #   "value": "1000000000000000000",
            #   "block_number": "12345678",
            #   "block_timestamp": "2023-01-01T00:00:00.000Z",
            #   "token_name": "Token",
            #   "token_symbol": "TKN",
            #   "token_decimals": "18"
            # }
            
            # Convert to Alchemy-like format for compatibility
            value_raw = moralis_transfer.get('value', '0')
            decimals = int(moralis_transfer.get('token_decimals', '18'))
            
            # Calculate human-readable value
            try:
                value_int = int(value_raw)
                value_float = value_int / (10 ** decimals)
            except (ValueError, TypeError):
                value_float = 0.0
            
            converted = {
                "hash": moralis_transfer.get('transaction_hash', ''),
                "from": moralis_transfer.get('from_address', ''),
                "to": moralis_transfer.get('to_address', ''),
                "value": str(value_float),
                "asset": moralis_transfer.get('token_symbol', 'UNKNOWN'),
                "blockNum": hex(int(moralis_transfer.get('block_number', '0'))),
                "rawContract": {
                    "address": moralis_transfer.get('address', ''),  # Token contract address
                    "decimals": decimals
                },
                "contractAddress": moralis_transfer.get('address', ''),
                "metadata": {
                    "blockTimestamp": moralis_transfer.get('block_timestamp', '')
                },
                # Additional Moralis-specific fields
                "token_name": moralis_transfer.get('token_name', ''),
                "token_decimals": str(decimals),
                "value_raw": value_raw
            }
            
            return converted
            
        except Exception as e:
            logger.error(f"Error converting Moralis format: {e}")
            return {}
    
    async def get_token_metadata(self, contract_address: str, network: str) -> Dict:
        """Get token metadata from Moralis"""
        try:
            chain = self._get_chain_id(network)
            url = f"{self.base_url}/erc20/metadata"
            
            headers = {
                "accept": "application/json",
                "X-API-Key": self.api_key
            }
            
            params = {
                "chain": chain,
                "addresses": [contract_address]
            }
            
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 0:
                            return data[0]
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting token metadata: {e}")
            return {}
    
    async def get_token_price(self, contract_address: str, network: str) -> Optional[float]:
        """Get token price from Moralis"""
        try:
            chain = self._get_chain_id(network)
            url = f"{self.base_url}/erc20/{contract_address}/price"
            
            headers = {
                "accept": "application/json",
                "X-API-Key": self.api_key
            }
            
            params = {
                "chain": chain
            }
            
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('usdPrice')
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting token price: {e}")
            return None
    
    async def get_wallet_token_balances(self, wallet_address: str, network: str) -> List[Dict]:
        """Get all ERC20 token balances for a wallet"""
        try:
            chain = self._get_chain_id(network)
            url = f"{self.base_url}/{wallet_address}/erc20"
            
            headers = {
                "accept": "application/json",
                "X-API-Key": self.api_key
            }
            
            params = {
                "chain": chain
            }
            
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting wallet balances: {e}")
            return []