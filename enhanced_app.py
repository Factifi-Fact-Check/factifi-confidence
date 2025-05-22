"""
Enhanced Claim Verification System with Real Source Integration
This system uses real APIs to gather evidence and implements proper mathematical confidence scoring.
"""

import os
import sys
import time
import json
import re
import traceback
import requests
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import concurrent.futures
import math
from urllib.parse import urlparse, quote

# Import LLM libraries
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

@dataclass
class SourceEvidence:
    """Represents evidence from a single source"""
    source_type: str  # 'academic', 'web', 'news', 'expert'
    credibility_score: float  # 0-1
    supports_claim: Optional[bool]  # True/False/None for can't determine
    content: str
    url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    citation_count: Optional[int] = None

@dataclass
class ConfidenceComponents:
    n_sources: int
    c_credibility: float
    a_agreement: float
    h_hedging: float
    t_tone: float
    evidence_score: float
    tone_score: float
    final_confidence: int
    n_max: int = 10

class RealSourceGatherer:
    """Real source gatherer using multiple APIs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ClaimVerifier/1.0 (Research Tool)'
        })
    
    def assess_domain_credibility(self, url: str) -> float:
        """Assess credibility based on domain reputation"""
        if not url:
            return 0.3
            
        domain = urlparse(url).netloc.lower()
        
        # Tier 1: Highest credibility (0.9-1.0)
        tier1_domains = {
            'nature.com': 0.98, 'science.org': 0.98, 'cell.com': 0.97,
            'nejm.org': 0.98, 'thelancet.com': 0.97, 'bmj.com': 0.96,
            'pnas.org': 0.97, 'ieee.org': 0.95, 'acm.org': 0.95,
            'springer.com': 0.93, 'elsevier.com': 0.93, 'wiley.com': 0.92,
            'pubmed.ncbi.nlm.nih.gov': 0.96, 'scholar.google.com': 0.90
        }
        
        # Tier 2: High credibility (0.75-0.89)
        tier2_domains = {
            'reuters.com': 0.88, 'bbc.com': 0.87, 'npr.org': 0.86,
            'pbs.org': 0.85, 'ap.org': 0.87, 'economist.com': 0.85,
            'ft.com': 0.84, 'wsj.com': 0.83, 'nytimes.com': 0.82,
            'washingtonpost.com': 0.81, 'theguardian.com': 0.80,
            'arxiv.org': 0.85, 'jstor.org': 0.88, 'researchgate.net': 0.78
        }
        
        # Tier 3: Moderate credibility (0.55-0.74)
        tier3_domains = {
            'wikipedia.org': 0.65, 'britannica.com': 0.72,
            'cnn.com': 0.68, 'foxnews.com': 0.60, 'msnbc.com': 0.62,
            'usatoday.com': 0.65, 'time.com': 0.68, 'newsweek.com': 0.66,
            'thehill.com': 0.63, 'politico.com': 0.67, 'axios.com': 0.70,
            'gov': 0.75  # Government domains
        }
        
        # Tier 4: Lower credibility (0.35-0.54)
        tier4_domains = {
            'medium.com': 0.45, 'wordpress.com': 0.40, 'blogspot.com': 0.35,
            'quora.com': 0.42, 'reddit.com': 0.38, 'facebook.com': 0.30,
            'twitter.com': 0.35, 'youtube.com': 0.40, 'tiktok.com': 0.25
        }
        
        # Check exact matches first
        for tier_domains in [tier1_domains, tier2_domains, tier3_domains, tier4_domains]:
            if domain in tier_domains:
                return tier_domains[domain]
        
        # Check for partial matches (e.g., subdomains)
        for tier_domains in [tier1_domains, tier2_domains, tier3_domains, tier4_domains]:
            for trusted_domain, score in tier_domains.items():
                if trusted_domain in domain or domain in trusted_domain:
                    return score * 0.9  # Slight penalty for partial match
        
        # Special handling for government and educational domains
        if domain.endswith('.gov') or domain.endswith('.edu'):
            return 0.75
        elif domain.endswith('.org'):
            return 0.60
        elif domain.endswith('.ac.uk') or domain.endswith('.edu.au'):
            return 0.80
            
        # Default for unknown domains
        return 0.50
    
    def search_semantic_scholar(self, query: str, limit: int = 5) -> List[SourceEvidence]:
        """Search Semantic Scholar for academic papers"""
        sources = []
        
        if not SEMANTIC_SCHOLAR_API_KEY:
            return sources
            
        try:
            headers = {'x-api-key': SEMANTIC_SCHOLAR_API_KEY}
            
            # Search for papers
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,year,citationCount,abstract,url,venue'
            }
            
            response = self.session.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for paper in data.get('data', []):
                    # Determine support based on abstract analysis
                    abstract = paper.get('abstract', '')
                    supports_claim = self._analyze_text_support(abstract, query) if abstract else None
                    
                    # Calculate credibility based on citations and venue
                    citation_count = paper.get('citationCount', 0)
                    venue = paper.get('venue', {})
                    base_credibility = 0.85  # High base for academic papers
                    
                    # Boost for high citations
                    if citation_count > 100:
                        base_credibility = min(0.95, base_credibility + 0.05)
                    elif citation_count > 50:
                        base_credibility = min(0.90, base_credibility + 0.03)
                    
                    # Boost for prestigious venues
                    venue_name = venue.get('name', '').lower() if isinstance(venue, dict) else str(venue).lower()
                    if any(prestigious in venue_name for prestigious in ['nature', 'science', 'cell', 'lancet']):
                        base_credibility = min(0.98, base_credibility + 0.08)
                    
                    authors = paper.get('authors', [])
                    author_names = ', '.join([author.get('name', '') for author in authors[:3]])
                    
                    sources.append(SourceEvidence(
                        source_type='academic',
                        credibility_score=base_credibility,
                        supports_claim=supports_claim,
                        content=abstract[:500] if abstract else paper.get('title', ''),
                        url=paper.get('url'),
                        title=paper.get('title'),
                        author=author_names,
                        publication_date=str(paper.get('year', '')),
                        citation_count=citation_count
                    ))
                    
        except Exception as e:
            print(f"Error searching Semantic Scholar: {str(e)}")
            
        return sources
    
    def search_web_serper(self, query: str, limit: int = 8) -> List[SourceEvidence]:
        """Search web using Serper API"""
        sources = []
        
        if not SERPER_API_KEY:
            return sources
            
        try:
            headers = {
                'X-API-KEY': SERPER_API_KEY,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'q': query,
                'num': limit,
                'hl': 'en',
                'gl': 'us'
            }
            
            response = self.session.post(
                'https://google.serper.dev/search',
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for result in data.get('organic', []):
                    title = result.get('title', '')
                    snippet = result.get('snippet', '')
                    url = result.get('link', '')
                    
                    # Determine support based on snippet analysis
                    supports_claim = self._analyze_text_support(snippet, query)
                    
                    # Assess credibility based on domain
                    credibility = self.assess_domain_credibility(url)
                    
                    # Determine source type based on domain
                    domain = urlparse(url).netloc.lower()
                    if any(news_domain in domain for news_domain in [
                        'reuters', 'bbc', 'npr', 'pbs', 'ap.org', 'economist'
                    ]):
                        source_type = 'news'
                    elif domain.endswith('.gov') or domain.endswith('.edu'):
                        source_type = 'official'
                    else:
                        source_type = 'web'
                    
                    sources.append(SourceEvidence(
                        source_type=source_type,
                        credibility_score=credibility,
                        supports_claim=supports_claim,
                        content=snippet,
                        url=url,
                        title=title
                    ))
                    
        except Exception as e:
            print(f"Error searching with Serper: {str(e)}")
            
        return sources
    
    def search_perplexity(self, query: str) -> Optional[SourceEvidence]:
        """Get expert analysis from Perplexity AI"""
        if not PERPLEXITY_API_KEY:
            return None
            
        try:
            headers = {
                'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'llama-3.1-sonar-small-128k-online',
                'messages': [
                    {
                        'role': 'user',
                        'content': f'Analyze this claim with current evidence and expert consensus: "{query}". Provide factual analysis with sources.'
                    }
                ],
                'max_tokens': 1000,
                'temperature': 0.1
            }
            
            response = self.session.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                # Analyze the expert response for support
                supports_claim = self._analyze_text_support(content, query)
                
                return SourceEvidence(
                    source_type='expert',
                    credibility_score=0.82,  # High credibility for AI expert analysis
                    supports_claim=supports_claim,
                    content=content[:800],
                    title="Expert AI Analysis",
                    author="Perplexity AI"
                )
                
        except Exception as e:
            print(f"Error with Perplexity search: {str(e)}")
            
        return None
    
    def _analyze_text_support(self, text: str, claim: str) -> Optional[bool]:
        """Analyze if text supports, opposes, or is neutral about claim"""
        if not text or len(text) < 20:
            return None
            
        text_lower = text.lower()
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        claim_words = set(re.findall(r'\b\w+\b', claim_lower))
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'must'}
        claim_words = claim_words - stop_words
        
        # Count relevant words in text
        relevant_word_count = sum(1 for word in claim_words if word in text_lower)
        relevance_score = relevant_word_count / len(claim_words) if claim_words else 0
        
        # If not relevant enough, return None
        if relevance_score < 0.3:
            return None
        
        # Look for supporting language
        support_indicators = [
            'confirm', 'support', 'evidence shows', 'research indicates', 'studies show',
            'proven', 'demonstrated', 'established', 'validates', 'corroborates',
            'according to', 'research suggests', 'findings show', 'data indicates'
        ]
        
        # Look for opposing language
        oppose_indicators = [
            'disprove', 'refute', 'contradict', 'false', 'incorrect', 'myth',
            'debunked', 'no evidence', 'not supported', 'lacks evidence',
            'studies show otherwise', 'contrary to', 'however', 'but research shows'
        ]
        
        # Look for uncertainty language
        uncertainty_indicators = [
            'unclear', 'mixed evidence', 'conflicting', 'debate', 'controversial',
            'some studies', 'limited evidence', 'more research needed', 'inconclusive'
        ]
        
        support_count = sum(1 for indicator in support_indicators if indicator in text_lower)
        oppose_count = sum(1 for indicator in oppose_indicators if indicator in text_lower)
        uncertain_count = sum(1 for indicator in uncertainty_indicators if indicator in text_lower)
        
        # Determine support based on indicators
        if support_count > oppose_count and support_count > uncertain_count:
            return True
        elif oppose_count > support_count and oppose_count > uncertain_count:
            return False
        else:
            return None  # Neutral or unclear
    
    def gather_evidence(self, claim: str) -> List[SourceEvidence]:
        """Gather evidence from multiple sources"""
        all_sources = []
        
        print(f"Gathering evidence for: {claim[:60]}...")
        
        # Search academic sources
        try:
            academic_sources = self.search_semantic_scholar(claim, limit=3)
            all_sources.extend(academic_sources)
            print(f"Found {len(academic_sources)} academic sources")
        except Exception as e:
            print(f"Academic search failed: {str(e)}")
        
        # Search web sources
        try:
            web_sources = self.search_web_serper(claim, limit=5)
            all_sources.extend(web_sources)
            print(f"Found {len(web_sources)} web sources")
        except Exception as e:
            print(f"Web search failed: {str(e)}")
        
        # Get expert analysis
        try:
            expert_source = self.search_perplexity(claim)
            if expert_source:
                all_sources.append(expert_source)
                print("Added expert analysis")
        except Exception as e:
            print(f"Expert analysis failed: {str(e)}")
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_sources = []
        
        for source in all_sources:
            url_key = source.url or f"{source.title}_{source.content[:50]}"
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique_sources.append(source)
        
        print(f"Total unique sources found: {len(unique_sources)}")
        return unique_sources

class EnhancedConfidenceCalculator:
    """Enhanced confidence calculator with proper mathematical implementation"""
    
    HEDGING_WORDS = [
        "may", "might", "could", "possibly", "likely", "suggest", 
        "appear", "indicate", "perhaps", "probably", "potentially",
        "seems", "appears", "allegedly", "reportedly", "supposedly",
        "presumably", "apparently", "ostensibly", "purportedly",
        "unclear", "uncertain", "ambiguous", "debatable", "questionable"
    ]
    
    CONFIDENCE_BOUNDARIES = {
        (0, 20): "Very Low - Highly unreliable",
        (21, 40): "Low - Questionable reliability", 
        (41, 60): "Moderate - Ambiguous evidence",
        (61, 80): "High - Generally reliable",
        (81, 100): "Very High - Highly reliable"
    }
    
    def __init__(self, n_max: int = 8, w_n: float = 0.25, w_c: float = 0.45, w_a: float = 0.30):
        """Initialize with proper weight distribution"""
        self.n_max = n_max
        self.w_n = w_n
        self.w_c = w_c  
        self.w_a = w_a
        
        # Normalize weights
        total_weight = w_n + w_c + w_a
        if abs(total_weight - 1.0) > 0.001:
            self.w_n = w_n / total_weight
            self.w_c = w_c / total_weight
            self.w_a = w_a / total_weight
    
    def calculate_hedging_density(self, text: str) -> float:
        """Calculate hedging word density with improved detection"""
        if not text or not text.strip():
            return 0.0
            
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
            
        # Count hedging words with context sensitivity
        hedge_count = 0
        for word in words:
            if word in self.HEDGING_WORDS:
                hedge_count += 1
        
        # Calculate density with smoothing
        density = hedge_count / len(words)
        
        # Apply ceiling to prevent over-penalization
        return min(density, 0.15)  # Cap at 15% to avoid extreme penalties
    
    def map_source_volume(self, n: int) -> float:
        """Map source count with improved scaling"""
        if n <= 0:
            return 0.0
        if n >= self.n_max:
            return 1.0
        
        # Use square root scaling for better distribution
        return math.sqrt(n) / math.sqrt(self.n_max)
    
    def calculate_confidence(self, sources: List[SourceEvidence], llm_response: str) -> ConfidenceComponents:
        """Calculate confidence with improved mathematical model"""
        
        # Handle empty sources
        if not sources:
            h = self.calculate_hedging_density(llm_response)
            t = 1.0 - h
            # Lower base confidence when no evidence
            base_confidence = max(30, int(t * 60))  # 30-60% range based on tone
            
            return ConfidenceComponents(
                n_sources=0,
                c_credibility=0.0,
                a_agreement=0.5,
                h_hedging=h,
                t_tone=t,
                evidence_score=0.0,
                tone_score=t,
                final_confidence=base_confidence
            )
        
        # Calculate components
        n = len(sources)
        n_mapped = self.map_source_volume(n)
        
        # Calculate weighted credibility (higher weight for supporting sources)
        credibility_scores = []
        weights = []
        
        for source in sources:
            if source.credibility_score is not None:
                credibility_scores.append(source.credibility_score)
                # Weight supporting/opposing sources more than neutral
                if source.supports_claim is not None:
                    weights.append(1.2)  # 20% more weight for decisive sources
                else:
                    weights.append(1.0)
        
        if credibility_scores:
            weighted_credibility = sum(c * w for c, w in zip(credibility_scores, weights)) / sum(weights)
            c = weighted_credibility
        else:
            c = 0.5
        
        # Calculate agreement with improved logic
        supporting = sum(1 for s in sources if s.supports_claim is True)
        opposing = sum(1 for s in sources if s.supports_claim is False)
        neutral = sum(1 for s in sources if s.supports_claim is None)
        
        total_sources = len(sources)
        decisive_sources = supporting + opposing
        
        if decisive_sources == 0:
            # All sources are neutral - moderate confidence reduction
            if neutral > 3:  # Many neutral sources suggest complexity
                a = 0.4
            else:
                a = 0.5
        else:
            # Calculate agreement proportion
            if supporting >= opposing:
                a = supporting / total_sources
                # Bonus for strong consensus
                if supporting > 0 and opposing == 0 and supporting >= 2:
                    a = min(1.0, a + 0.1)
            else:
                # More sources oppose than support
                a = 0.3 - (opposing / total_sources * 0.2)  # Lower confidence
                a = max(0.1, a)  # Floor at 0.1
        
        # Calculate hedging and tone
        h = self.calculate_hedging_density(llm_response)
        t = 1.0 - h
        
        # Calculate evidence score with improved weighting
        evidence_score = (self.w_n * n_mapped) + (self.w_c * c) + (self.w_a * a)
        
        # Apply source quality bonus
        high_quality_sources = sum(1 for s in sources if s.credibility_score > 0.8)
        if high_quality_sources >= 2:
            evidence_score = min(1.0, evidence_score + 0.05)
        
        # Calculate final confidence with adjusted weighting
        # Give more weight to evidence when we have good sources
        if n >= 3 and c > 0.7:
            # Strong evidence scenario
            final_score = (0.8 * evidence_score) + (0.2 * t)
        elif n >= 1 and c > 0.5:
            # Moderate evidence scenario
            final_score = (0.7 * evidence_score) + (0.3 * t)
        else:
            # Weak evidence scenario
            final_score = (0.5 * evidence_score) + (0.5 * t)
        
        # Convert to percentage with proper bounds
        final_confidence = int(final_score * 100)
        final_confidence = max(15, min(95, final_confidence))  # Reasonable bounds
        
        return ConfidenceComponents(
            n_sources=n,
            c_credibility=c,
            a_agreement=a,
            h_hedging=h,
            t_tone=t,
            evidence_score=evidence_score,
            tone_score=t,
            final_confidence=final_confidence,
            n_max=self.n_max
        )
    
    def get_confidence_interpretation(self, confidence: int) -> str:
        """Get human-readable confidence interpretation"""
        for (min_conf, max_conf), interpretation in self.CONFIDENCE_BOUNDARIES.items():
            if min_conf <= confidence <= max_conf:
                return interpretation
        return "Unknown confidence level"

class EnhancedClaimVerifier:
    """Enhanced claim verifier with real source integration"""
    
    def __init__(self, llm):
        self.llm = llm
        self.source_gatherer = RealSourceGatherer()
        self.confidence_calc = EnhancedConfidenceCalculator()
        
        self.verification_prompt = ChatPromptTemplate.from_template("""
        You are an expert fact-checker. Analyze the following claim and evidence to provide a verification.

        CLAIM: {claim}

        EVIDENCE FROM SOURCES:
        {evidence}

        Your task:
        1. Determine if the claim is TRUE, FALSE, or if you CAN'T SAY based on the evidence
        2. Provide detailed reasoning
        3. Consider the credibility and consistency of sources
        4. Be precise about what the evidence shows vs. doesn't show

        Guidelines:
        - Use TRUE only if evidence strongly supports the claim
        - Use FALSE only if evidence clearly contradicts the claim  
        - Use CAN'T SAY if evidence is mixed, insufficient, or unclear
        - Avoid excessive hedging unless genuinely uncertain
        - Focus on the specific claim, not related topics

        Response format:
        VERDICT: [TRUE/FALSE/CAN'T SAY]
        REASONING: [Your detailed analysis explaining the verdict based on the evidence]
        """)
    
    def classify_claim(self, claim: str) -> str:
        """Classify claim type for better processing"""
        claim_lower = claim.lower()
        
        # Statistical/Research claims
        if any(word in claim_lower for word in [
            'study', 'research', 'scientist', 'percent', '%', 'survey', 
            'data shows', 'statistics', 'according to research'
        ]):
            return "Scientific/Statistical"
        
        # Quote/Attribution claims  
        elif any(word in claim_lower for word in [
            'said', 'stated', 'according to', 'quote', 'claimed', 'announced'
        ]):
            return "Quote/Statement"
        
        # Historical/Factual claims
        elif any(word in claim_lower for word in [
            'happened', 'occurred', 'when', 'date', 'year', 'during', 'was', 'were'
        ]):
            return "Historical/Factual"
        
        # Health/Medical claims
        elif any(word in claim_lower for word in [
            'health', 'medical', 'disease', 'treatment', 'cure', 'symptoms', 'doctor'
        ]):
            return "Health/Medical"
        
        # Future/Predictive claims
        elif any(word in claim_lower for word in [
            'will', 'going to', 'predict', 'future', 'forecast', 'expect'
        ]):
            return "Predictive/Future"
        
        else:
            return "General Factual"
    
    def format_evidence(self, sources: List[SourceEvidence]) -> str:
        """Format evidence for LLM prompt"""
        if not sources:
            return "No external evidence sources found. Analysis based on general knowledge only."
        
        evidence_text = ""
        for i, source in enumerate(sources, 1):
            support_text = {
                True: "SUPPORTS the claim",
                False: "OPPOSES the claim", 
                None: "NEUTRAL/UNCLEAR on the claim"
            }.get(source.supports_claim, "UNKNOWN")
            
            evidence_text += f"""
Source {i} - {source.source_type.upper()}:
Title: {source.title or 'Unknown'}
Author: {source.author or 'Unknown'}
URL: {source.url or 'Not available'}
Credibility Score: {source.credibility_score:.2f}/1.0
Position on Claim: {support_text}
Content: {source.content[:300]}...
{f"Citations: {source.citation_count}" if source.citation_count else ""}

"""
        
        return evidence_text.strip()
    
    def verify_claim(self, claim: str) -> Dict:
        """Verify claim with real source gathering"""
        try:
            print(f"\n=== Verifying: {claim[:80]}... ===")
            
            # Classify claim
            claim_type = self.classify_claim(claim)
            print(f"Claim type: {claim_type}")
            
            # Gather real evidence
            sources = self.source_gatherer.gather_evidence(claim)
            
            # Format evidence
            evidence_text = self.format_evidence(sources)
            
            # Get LLM verification
            prompt = self.verification_prompt.format(
                claim=claim,
                evidence=evidence_text
            )
            
            response = self.llm.invoke(prompt).content
            
            # Parse verdict
            verdict = "CAN'T SAY"
            reasoning = response
            
            if "VERDICT:" in response:
                lines = response.split('\n')
                for line in lines:
                    if line.strip().startswith("VERDICT:"):
                        verdict_text = line.replace("VERDICT:", "").strip().upper()
                        if "TRUE" in verdict_text and "FALSE" not in verdict_text:
                            verdict = "TRUE"
                        elif "FALSE" in verdict_text:
                            verdict = "FALSE"
                        else:
                            verdict = "CAN'T SAY"
                        break
                
                # Extract reasoning
                reasoning_start = response.find("REASONING:")
                if reasoning_start != -1:
                    reasoning = response[reasoning_start + 10:].strip()
                else:
                    reasoning = response
            
            # Calculate confidence
            confidence_components = self.confidence_calc.calculate_confidence(sources, response)
            
            # Generate summary statistics
            source_breakdown = {}
            for source in sources:
                source_type = source.source_type
                source_breakdown[source_type] = source_breakdown.get(source_type, 0) + 1
            
            # Prepare result
            result = {
                'claim': claim,
                'claim_type': claim_type,
                'verdict': verdict,
                'reasoning': reasoning,
                'confidence_score': confidence_components.final_confidence,
                'confidence_interpretation': self.confidence_calc.get_confidence_interpretation(
                    confidence_components.final_confidence
                ),
                'sources_found': len(sources),
                'source_breakdown': source_breakdown,
                'confidence_components': {
                    'n_sources': confidence_components.n_sources,
                    'credibility_score': round(confidence_components.c_credibility, 3),
                    'agreement_score': round(confidence_components.a_agreement, 3),
                    'hedging_density': round(confidence_components.h_hedging, 3),
                    'tone_score': round(confidence_components.t_tone, 3),
                    'evidence_score': round(confidence_components.evidence_score, 3)
                },
                'sources': [
                    {
                        'type': s.source_type,
                        'title': s.title,
                        'url': s.url,
                        'credibility': round(s.credibility_score, 2),
                        'supports_claim': s.supports_claim,
                        'content_preview': s.content[:150] + "..." if len(s.content) > 150 else s.content
                    } for s in sources
                ],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return result
            
        except Exception as e:
            print(f"Error verifying claim: {str(e)}")
            traceback.print_exc()
            
            return {
                'claim': claim,
                'verdict': 'ERROR',
                'reasoning': f'Verification failed due to error: {str(e)}',
                'confidence_score': 0,
                'confidence_interpretation': 'Error - Could not verify',
                'sources_found': 0,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

def setup_llm() -> Any:
    """Setup and return configured LLM"""
    if OPENAI_API_KEY:
        print("Using OpenAI GPT-4")
        return ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=OPENAI_API_KEY
        )
    elif GEMINI_API_KEY:
        print("Using Google Gemini")
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.1,
            google_api_key=GEMINI_API_KEY
        )
    else:
        raise ValueError("No LLM API key found. Please set OPENAI_API_KEY or GEMINI_API_KEY")

def print_verification_result(result: Dict):
    """Print formatted verification result"""
    print("\n" + "="*80)
    print(f"CLAIM VERIFICATION REPORT")
    print("="*80)
    
    print(f"Claim: {result['claim']}")
    print(f"Type: {result.get('claim_type', 'Unknown')}")
    print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
    
    print(f"\nVERDICT: {result['verdict']}")
    print(f"Confidence: {result['confidence_score']}% ({result.get('confidence_interpretation', 'Unknown')})")
    
    print(f"\nREASONING:")
    print(result['reasoning'])
    
    if result.get('sources_found', 0) > 0:
        print(f"\nSOURCE SUMMARY:")
        print(f"Total sources found: {result['sources_found']}")
        
        breakdown = result.get('source_breakdown', {})
        if breakdown:
            print("Source types:")
            for source_type, count in breakdown.items():
                print(f"  - {source_type.title()}: {count}")
        
        print(f"\nCONFIDENCE BREAKDOWN:")
        components = result.get('confidence_components', {})
        if components:
            print(f"  - Source count score: {components.get('n_sources', 0)}")
            print(f"  - Average credibility: {components.get('credibility_score', 0):.3f}")
            print(f"  - Source agreement: {components.get('agreement_score', 0):.3f}")
            print(f"  - Hedging density: {components.get('hedging_density', 0):.3f}")
            print(f"  - Tone confidence: {components.get('tone_score', 0):.3f}")
            print(f"  - Overall evidence score: {components.get('evidence_score', 0):.3f}")
        
        print(f"\nSOURCE DETAILS:")
        sources = result.get('sources', [])
        for i, source in enumerate(sources, 1):
            support_text = {
                True: "SUPPORTS",
                False: "OPPOSES", 
                None: "NEUTRAL"
            }.get(source.get('supports_claim'), "UNKNOWN")
            
            print(f"{i}. [{source.get('type', 'unknown').upper()}] {source.get('title', 'Unknown Title')}")
            print(f"   Credibility: {source.get('credibility', 0):.2f}/1.0 | Position: {support_text}")
            if source.get('url'):
                print(f"   URL: {source['url']}")
            print(f"   Preview: {source.get('content_preview', 'No preview available')}")
            print()
    else:
        print(f"\nNo external sources found. Analysis based on general knowledge.")
    
    print("="*80)

def main():
    """Main function for interactive claim verification"""
    try:
        # Setup LLM
        llm = setup_llm()
        
        # Initialize verifier
        verifier = EnhancedClaimVerifier(llm)
        
        print("Enhanced Claim Verification System")
        print("=" * 50)
        print("Enter claims to verify (type 'quit' to exit)")
        print("API Keys detected:", {
            'OpenAI': bool(OPENAI_API_KEY),
            'Gemini': bool(GEMINI_API_KEY), 
            'Semantic Scholar': bool(SEMANTIC_SCHOLAR_API_KEY),
            'Serper': bool(SERPER_API_KEY),
            'Perplexity': bool(PERPLEXITY_API_KEY),
            'Jina': bool(JINA_API_KEY)
        })
        
        while True:
            try:
                claim = input("\nEnter claim to verify: ").strip()
                
                if not claim:
                    continue
                    
                if claim.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Verify claim
                start_time = time.time()
                result = verifier.verify_claim(claim)
                end_time = time.time()
                
                # Add processing time
                result['processing_time'] = f"{end_time - start_time:.2f} seconds"
                
                # Print results
                print_verification_result(result)
                
                # Optionally save to file
                save_option = input("\nSave result to JSON file? (y/n): ").strip().lower()
                if save_option == 'y':
                    filename = f"verification_{int(time.time())}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"Result saved to {filename}")
                
            except KeyboardInterrupt:
                print("\n\nProcess interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError processing claim: {str(e)}")
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"Failed to initialize system: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

def verify_single_claim(claim: str) -> Dict:
    """Verify a single claim and return result (for programmatic use)"""
    try:
        llm = setup_llm()
        verifier = EnhancedClaimVerifier(llm)
        return verifier.verify_claim(claim)
    except Exception as e:
        return {
            'claim': claim,
            'verdict': 'ERROR',
            'reasoning': f'Verification failed: {str(e)}',
            'confidence_score': 0,
            'error': str(e)
        }

def batch_verify_claims(claims: List[str]) -> List[Dict]:
    """Verify multiple claims in batch"""
    results = []
    
    try:
        llm = setup_llm()
        verifier = EnhancedClaimVerifier(llm)
        
        for i, claim in enumerate(claims, 1):
            print(f"\nProcessing claim {i}/{len(claims)}: {claim[:60]}...")
            try:
                result = verifier.verify_claim(claim)
                results.append(result)
                
                # Brief delay to respect API limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing claim {i}: {str(e)}")
                results.append({
                    'claim': claim,
                    'verdict': 'ERROR', 
                    'reasoning': f'Processing failed: {str(e)}',
                    'confidence_score': 0,
                    'error': str(e)
                })
                
    except Exception as e:
        print(f"Batch processing failed: {str(e)}")
        
    return results

# Example usage and test cases
if __name__ == "__main__":
    # Check if running as script
    if len(sys.argv) > 1:
        # Command line usage
        claim = " ".join(sys.argv[1:])
        result = verify_single_claim(claim)
        print_verification_result(result)
    else:
        # Interactive mode
        sys.exit(main())

# Test claims for development
TEST_CLAIMS = [
    "Coffee reduces the risk of type 2 diabetes",
    "The Great Wall of China is visible from space with the naked eye",
    "Vaccines cause autism in children",
    "Climate change is primarily caused by human activities",
    "Einstein said 'Imagination is more important than knowledge'",
    "Drinking 8 glasses of water per day is necessary for health",
    "The human brain only uses 10% of its capacity",
    "Goldfish have a 3-second memory span"
]

def run_tests():
    """Run test verification on sample claims"""
    print("Running test verifications...")
    results = batch_verify_claims(TEST_CLAIMS[:3])  # Test first 3 claims
    
    for result in results:
        print_verification_result(result)
        print("\n" + "-"*50 + "\n")

# Usage examples:
# python claim_verifier.py "Coffee reduces diabetes risk"
# python claim_verifier.py  # Interactive mode
