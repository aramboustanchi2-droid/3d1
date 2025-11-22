"""
University Knowledge Integration - ادغام دانش دانشگاهی با RAG

اتصال ایجنت‌های دانشگاهی به سیستم RAG برای یادگیری مداوم
"""

import io
import sys
import os
import logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cad3d.super_ai.university_config import UNIVERSITIES, AGENT_CONFIG
from cad3d.super_ai.university_agents import UniversityAgentManager
from cad3d.super_ai.rag_system import RAGSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

class UniversityKnowledgeIntegration:
    """
    سیستم یکپارچه برای یادگیری از دانشگاه‌ها و ادغام با RAG
    """
    
    def __init__(self):
        logger.info("\n" + "="*80)
        logger.info("  UNIVERSITY KNOWLEDGE INTEGRATION SYSTEM")
        logger.info("  سیستم یادگیری از دانشگاه‌های برتر دنیا")
        logger.info("="*80 + "\n")
        
        # Initialize RAG system
        logger.info("Step 1: Initializing RAG System...")
        self.rag_system = RAGSystem()
        logger.info("✓ RAG System ready\n")
        
        # Initialize agent manager
        logger.info("Step 2: Initializing University Agents...")
        self.agent_manager = UniversityAgentManager(
            UNIVERSITIES,
            AGENT_CONFIG,
            self.rag_system
        )
        logger.info("✓ All agents ready\n")
    
    def show_universities(self):
        """نمایش لیست دانشگاه‌ها"""
        logger.info("\n" + "="*80)
        logger.info("  AVAILABLE UNIVERSITIES")
        logger.info("="*80 + "\n")
        
        for i, (key, info) in enumerate(UNIVERSITIES.items(), 1):
            logger.info(f"{i}. {info['name']}")
            logger.info(f"   Country: {info['country']}")
            logger.info(f"   Rank: #{info['rank']}")
            logger.info(f"   Focus: {', '.join(info['focus_areas'][:3])}")
            logger.info(f"   Resources: {len(info['resources'])} sources")
            logger.info("")
    
    def demo_single_university(self, university_key: str = "MIT"):
        """
        دمو: یادگیری از یک دانشگاه
        
        Args:
            university_key: کلید دانشگاه (مثلا "MIT")
        """
        logger.info("\n" + "="*80)
        logger.info(f"  DEMO: Learning from {UNIVERSITIES[university_key]['name']}")
        logger.info("="*80 + "\n")
        
        # یادگیری
        result = self.agent_manager.learn_from_specific([university_key])
        
        # نمایش نتیجه
        if result['results']:
            stats = result['results'][0]
            logger.info("\n  Results:")
            logger.info(f"  Status: {stats['status']}")
            logger.info(f"  Pages scraped: {stats['pages_scraped']}")
            logger.info(f"  Documents processed: {stats['documents_processed']}")
            logger.info(f"  Documents added to RAG: {stats['documents_added_to_rag']}")
        
        return result
    
    def learn_from_top_5(self):
        """یادگیری از 5 دانشگاه برتر"""
        logger.info("\n" + "="*80)
        logger.info("  Learning from Top 5 Universities")
        logger.info("="*80 + "\n")
        
        top_5 = ["MIT", "Stanford", "Cambridge", "Oxford", "Berkeley"]
        result = self.agent_manager.learn_from_specific(top_5)
        
        logger.info("\n  Summary:")
        logger.info(f"  Agents updated: {result['agents_updated']}")
        logger.info(f"  Total documents: {sum(r['documents_added_to_rag'] for r in result['results'])}")
        
        return result
    
    def learn_from_all(self):
        """یادگیری از همه دانشگاه‌ها"""
        logger.info("\n" + "="*80)
        logger.info("  Learning from ALL Universities")
        logger.info("="*80 + "\n")
        
        return self.agent_manager.learn_from_all()
    
    def show_statistics(self):
        """نمایش آمار سیستم"""
        stats = self.agent_manager.get_statistics()
        
        logger.info("\n" + "="*80)
        logger.info("  SYSTEM STATISTICS")
        logger.info("="*80 + "\n")
        
        logger.info(f"Total Agents: {stats['total_agents']}")
        logger.info(f"Agents Needing Update: {stats['agents_needing_update']}")
        logger.info(f"Total Documents Collected: {stats['total_documents_collected']}")
        logger.info(f"Total Pages Scraped: {stats['total_pages_scraped']}")
        
        logger.info("\n  By University:")
        for uni, data in stats['by_university'].items():
            logger.info(f"\n  {uni}:")
            logger.info(f"    Country: {data['country']}")
            logger.info(f"    Documents: {data['documents']}")
            logger.info(f"    Pages: {data['pages']}")
            logger.info(f"    Last Update: {data['last_update'] or 'Never'}")
        
        return stats
    
    def test_rag_query(self, query: str):
        """
        تست پرسش از RAG با دانش دانشگاهی
        
        Args:
            query: پرسش
        """
        logger.info("\n" + "="*80)
        logger.info(f"  Testing RAG Query")
        logger.info("="*80 + "\n")
        
        logger.info(f"Query: {query}")
        
        # پرسش از RAG
        results = self.rag_system.search(query, top_k=3)
        
        logger.info(f"\n  Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            logger.info(f"\n  {i}. Score: {result['score']:.4f}")
            logger.info(f"     University: {result['metadata'].get('university', 'Unknown')}")
            logger.info(f"     Resource: {result['metadata'].get('resource', 'Unknown')}")
            logger.info(f"     URL: {result['metadata'].get('url', 'Unknown')}")
            logger.info(f"     Content preview: {result['content'][:200]}...")
        
        return results


def main():
    """اجرای دمو"""
    
    # Initialize system
    system = UniversityKnowledgeIntegration()
    
    # نمایش دانشگاه‌ها
    system.show_universities()
    
    # دمو: یادگیری از MIT
    logger.info("\n" + "="*80)
    logger.info("  DEMO MODE: Quick Test with Limited Data")
    logger.info("  (For full collection, run learn_from_all())")
    logger.info("="*80 + "\n")
    
    # یادگیری از MIT (محدود)
    result = system.demo_single_university("MIT")
    
    # نمایش آمار
    system.show_statistics()
    
    # تست پرسش
    if result['results'] and result['results'][0]['documents_added_to_rag'] > 0:
        system.test_rag_query("artificial intelligence research")
    
    logger.info("\n" + "="*80)
    logger.info("  INTEGRATION COMPLETE")
    logger.info("="*80 + "\n")
    
    logger.info("Next Steps:")
    logger.info("  1. Run learn_from_top_5() for more universities")
    logger.info("  2. Run learn_from_all() for complete collection")
    logger.info("  3. Set up scheduler for automatic updates")
    logger.info("  4. Integrate with UnifiedAISystem for queries")
    logger.info("")


if __name__ == "__main__":
    main()
