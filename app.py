"""
Enhanced Claim Verification System with Mathematical Confidence Scoring
This system implements a rigorous mathematical approach to calculating confidence scores
based on evidence volume, source credibility, agreement proportion, and hedging analysis.
"""

import os
import sys
import time
import json
import re
import traceback
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import concurrent.futures
import math

# Import LLM libraries
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment variables")

@dataclass
class SourceEvidence:
    """Represents evidence from a single source"""
    source_type: str  # 'academic', 'web', 'patent', 'review'
    credibility_score: float  # 0-1
    supports_claim: Optional[bool]  # True/False/None for can't determine
    content: str
    url: Optional[str] = None
    title: Optional[str] = None

@dataclass
class ConfidenceComponents:
    """Components used in confidence calculation"""
    n_sources: int  # Number of distinct evidence sources
    n_max: int = 10  # Maximum source threshold
    c_credibility: float  # Mean credibility score
    a_agreement: float  # Agreement proportion (0-1)
    h_hedging: float  # Hedging proportion (0-1)
    t_tone: float  # Tone factor (1-H)
    evidence_score: float  # Combined evidence score
    tone_score: float  # Final tone score
    final_confidence: int  # Final confidence percentage

class EnhancedConfidenceCalculator:
    """Enhanced confidence calculator with mathematical rigor"""
    
    # Hedging words for uncertainty detection
    HEDGING_WORDS = [
        "may", "might", "could", "possibly", "likely", "suggest", 
        "appear", "indicate", "perhaps", "probably", "potentially",
        "seems", "appears", "allegedly", "reportedly", "supposedly",
        "presumably", "apparently", "ostensibly", "purportedly"
    ]
    
    # Confidence interpretation boundaries
    CONFIDENCE_BOUNDARIES = {
        (0, 20): "Very Low - Highly unreliable",
        (21, 40): "Low - Questionable reliability", 
        (41, 60): "Moderate - Ambiguous evidence",
        (61, 80): "High - Generally reliable",
        (81, 100): "Very High - Highly reliable"
    }
    
    def __init__(self, n_max: int = 10, w_n: float = 0.2, w_c: float = 0.5, w_a: float = 0.3):
        """
        Initialize confidence calculator with weights
        
        Args:
            n_max: Maximum source count for diminishing returns
            w_n: Weight for source volume (0-1)
            w_c: Weight for credibility (0-1) 
            w_a: Weight for agreement (0-1)
        """
        self.n_max = n_max
        self.w_n = w_n
        self.w_c = w_c  
        self.w_a = w_a
        
        # Ensure weights sum to 1
        total_weight = w_n + w_c + w_a
        if abs(total_weight - 1.0) > 0.001:
            print(f"WARNING: Evidence weights sum to {total_weight}, normalizing...")
            self.w_n = w_n / total_weight
            self.w_c = w_c / total_weight
            self.w_a = w_a / total_weight
    
    def calculate_hedging_density(self, text: str) -> float:
        """Calculate proportion of hedging words in text"""
        if not text or not text.strip():
            return 0.0
            
        # Clean and tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
            
        # Count hedging words
        hedge_count = sum(1 for word in words if word in self.HEDGING_WORDS)
        
        return hedge_count / len(words)
    
    def map_source_volume(self, n: int) -> float:
        """Map raw source count to 0-1 scale with diminishing returns"""
        if n <= 0:
            return 0.0
        if n >= self.n_max:
            return 1.0
        
        # Use logarithmic scaling for diminishing returns
        return math.log(n + 1) / math.log(self.n_max + 1)
    
    def calculate_confidence(self, sources: List[SourceEvidence], llm_response: str) -> ConfidenceComponents:
        """
        Calculate confidence score using mathematical approach
        
        Args:
            sources: List of evidence sources
            llm_response: LLM's verification response text
            
        Returns:
            ConfidenceComponents with detailed breakdown
        """
        # Handle no evidence case
        if not sources:
            return ConfidenceComponents(
                n_sources=0,
                n_max=self.n_max,
                c_credibility=0.0,
                a_agreement=0.5,  # Neutral when no evidence
                h_hedging=self.calculate_hedging_density(llm_response),
                t_tone=1.0 - self.calculate_hedging_density(llm_response),
                evidence_score=0.0,
                tone_score=1.0 - self.calculate_hedging_density(llm_response),
                final_confidence=50  # Default fallback
            )
        
        # Calculate components
        n = len(sources)
        n_mapped = self.map_source_volume(n)
        
        # Calculate mean credibility
        credibility_scores = [s.credibility_score for s in sources if s.credibility_score is not None]
        c = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5
        
        # Calculate agreement proportion
        supporting = sum(1 for s in sources if s.supports_claim is True)
        opposing = sum(1 for s in sources if s.supports_claim is False)
        total_decisive = supporting + opposing
        
        if total_decisive == 0:
            a = 0.5  # Neutral when no decisive evidence
        else:
            a = supporting / total_decisive
        
        # Calculate hedging and tone
        h = self.calculate_hedging_density(llm_response)
        t = 1.0 - h
        
        # Calculate Evidence Score
        evidence_score = (self.w_n * n_mapped) + (self.w_c * c) + (self.w_a * a)
        
        # Calculate final confidence (Evidence Score 70%, Tone Score 30%)
        final_score = (0.7 * evidence_score) + (0.3 * t)
        final_confidence = round(final_score * 100)
        
        # Ensure confidence is within bounds
        final_confidence = max(0, min(100, final_confidence))
        
        return ConfidenceComponents(
            n_sources=n,
            n_max=self.n_max,
            c_credibility=c,
            a_agreement=a,
            h_hedging=h,
            t_tone=t,
            evidence_score=evidence_score,
            tone_score=t,
            final_confidence=final_confidence
        )
    
    def get_confidence_interpretation(self, confidence: int) -> str:
        """Get human-readable confidence interpretation"""
        for (min_conf, max_conf), interpretation in self.CONFIDENCE_BOUNDARIES.items():
            if min_conf <= confidence <= max_conf:
                return interpretation
        return "Unknown confidence level"

class MockSourceHandler:
    """Mock source handler for demonstration purposes"""
    
    def __init__(self):
        self.confidence_calc = EnhancedConfidenceCalculator()
    
    def assess_source_credibility(self, source_url: str, source_type: str = "web") -> float:
        """
        Mock credibility assessment - in real implementation this would:
        - Check domain reputation
        - Analyze author credentials  
        - Verify publication venue
        - Check citation metrics
        - Analyze user reviews/ratings
        """
        # Mock scoring based on common patterns
        if not source_url:
            return 0.3
            
        url_lower = source_url.lower()
        
        # High credibility sources
        if any(domain in url_lower for domain in [
            'nature.com', 'science.org', 'nejm.org', 'cell.com',
            'pnas.org', 'bmj.com', 'thelancet.com', 'ieee.org'
        ]):
            return 0.95
        
        # Medium-high credibility
        elif any(domain in url_lower for domain in [
            'pubmed', 'scholar.google', 'arxiv.org', 'jstor.org',
            'reuters.com', 'bbc.com', 'npr.org', 'pbs.org'
        ]):
            return 0.8
        
        # Medium credibility
        elif any(domain in url_lower for domain in [
            'wikipedia.org', 'britannica.com', 'nytimes.com',
            'washingtonpost.com', 'theguardian.com', 'cnn.com'
        ]):
            return 0.65
        
        # Lower credibility
        elif any(domain in url_lower for domain in [
            'blog', 'medium.com', 'quora.com', 'reddit.com'
        ]):
            return 0.4
        
        # Default for unknown sources
        return 0.5
    
    def gather_evidence(self, claim: str) -> List[SourceEvidence]:
        """
        Mock evidence gathering - in real implementation this would:
        - Search Semantic Scholar for academic papers
        - Search web for relevant articles
        - Check patent databases
        - Analyze user reviews and ratings
        - Gather expert opinions
        """
        # Simulate different types of evidence based on claim content
        sources = []
        
        # Mock academic source
        if "research" in claim.lower() or "study" in claim.lower():
            sources.append(SourceEvidence(
                source_type="academic",
                credibility_score=0.9,
                supports_claim=True,
                content=f"Academic research supports aspects of the claim: {claim[:100]}...",
                url="https://pubmed.ncbi.nlm.nih.gov/mock-study-123",
                title="Mock Academic Study on Related Topic"
            ))
        
        # Mock web sources
        sources.extend([
            SourceEvidence(
                source_type="web",
                credibility_score=self.assess_source_credibility("https://reuters.com/mock-article"),
                supports_claim=True,
                content=f"News article provides evidence for: {claim[:100]}...",
                url="https://reuters.com/mock-article",
                title="Mock News Article"
            ),
            SourceEvidence(
                source_type="web", 
                credibility_score=self.assess_source_credibility("https://bbc.com/mock-article"),
                supports_claim=None,  # Neutral/unclear
                content=f"Article mentions topic but is inconclusive: {claim[:100]}...",
                url="https://bbc.com/mock-article",
                title="Mock BBC Article"
            )
        ])
        
        # Sometimes add opposing evidence
        if "controversial" in claim.lower() or len(claim) > 100:
            sources.append(SourceEvidence(
                source_type="web",
                credibility_score=self.assess_source_credibility("https://nytimes.com/mock-counter"),
                supports_claim=False,
                content=f"Counter-evidence found: {claim[:100]}...",
                url="https://nytimes.com/mock-counter", 
                title="Mock Counter-Evidence Article"
            ))
        
        return sources

class EnhancedClaimVerifier:
    """Enhanced claim verifier with mathematical confidence scoring"""
    
    def __init__(self, llm):
        self.llm = llm
        self.source_handler = MockSourceHandler()
        self.confidence_calc = EnhancedConfidenceCalculator()
        
        # Verification prompt template
        self.verification_prompt = ChatPromptTemplate.from_template("""
        You are a fact-checking expert. Analyze the following claim and evidence to provide a verification.

        CLAIM: {claim}

        EVIDENCE:
        {evidence}

        Please provide:
        1. A clear TRUE/FALSE/CAN'T SAY verdict
        2. A detailed explanation of your reasoning
        3. How well the evidence supports or refutes the claim

        Be thorough but concise. Use hedge words (may, might, could, possibly, likely, suggest, appear, indicate) only when genuinely uncertain.

        Response format:
        VERDICT: [TRUE/FALSE/CAN'T SAY]
        EXPLANATION: [Your detailed analysis]
        """)
    
    def classify_claim(self, claim: str) -> str:
        """Classify claim type"""
        claim_lower = claim.lower()
        
        if any(word in claim_lower for word in ['study', 'research', 'scientist', 'percent', '%']):
            return "Scientific/Statistical"
        elif any(word in claim_lower for word in ['said', 'stated', 'according to', 'quote']):
            return "Quote/Statement"
        elif any(word in claim_lower for word in ['happened', 'occurred', 'when', 'date']):
            return "Historical/Factual"
        elif any(word in claim_lower for word in ['will', 'going to', 'predict', 'future']):
            return "Predictive/Future"
        else:
            return "General Factual"
    
    def format_evidence(self, sources: List[SourceEvidence]) -> str:
        """Format evidence sources for LLM prompt"""
        if not sources:
            return "No external evidence found. Relying on general knowledge."
        
        evidence_text = ""
        for i, source in enumerate(sources, 1):
            support_text = {
                True: "SUPPORTS",
                False: "OPPOSES", 
                None: "NEUTRAL/UNCLEAR"
            }.get(source.supports_claim, "UNKNOWN")
            
            evidence_text += f"""
            Source {i} ({source.source_type.upper()}):
            Title: {source.title or 'Unknown'}
            URL: {source.url or 'Unknown'}
            Credibility: {source.credibility_score:.2f}/1.0
            Position: {support_text}
            Content: {source.content[:200]}...
            
            """
        
        return evidence_text.strip()
    
    def verify_claim(self, claim: str) -> Dict:
        """Verify a single claim with enhanced confidence scoring"""
        try:
            print(f"Verifying claim: {claim[:100]}...")
            
            # Classify claim
            claim_type = self.classify_claim(claim)
            
            # Gather evidence
            sources = self.source_handler.gather_evidence(claim)
            
            # Format evidence for LLM
            evidence_text = self.format_evidence(sources)
            
            # Get LLM verification
            prompt = self.verification_prompt.format(
                claim=claim,
                evidence=evidence_text
            )
            
            response = self.llm.invoke(prompt).content
            
            # Parse verdict
            verdict = "CAN'T SAY"  # Default
            explanation = response
            
            if "VERDICT:" in response:
                lines = response.split('\n')
                for line in lines:
                    if line.startswith("VERDICT:"):
                        verdict_text = line.replace("VERDICT:", "").strip()
                        if "TRUE" in verdict_text.upper() and "FALSE" not in verdict_text.upper():
                            verdict = "TRUE"
                        elif "FALSE" in verdict_text.upper():
                            verdict = "FALSE"
                        else:
                            verdict = "CAN'T SAY"
                        break
                
                # Extract explanation
                if "EXPLANATION:" in response:
                    explanation = response.split("EXPLANATION:")[1].strip()
            
            # Calculate confidence with mathematical approach
            confidence_components = self.confidence_calc.calculate_confidence(sources, response)
            
            # Determine method
            method = "enhanced_evidence" if sources else "llm_knowledge"
            
            # Get evidence source types
            evidence_sources = list(set(s.source_type for s in sources)) if sources else []
            
            return {
                "claim": claim,
                "type": claim_type,
                "verdict": verdict,
                "confidence": confidence_components.final_confidence,
                "confidence_interpretation": self.confidence_calc.get_confidence_interpretation(
                    confidence_components.final_confidence
                ),
                "confidence_components": {
                    "n_sources": confidence_components.n_sources,
                    "n_mapped": self.confidence_calc.map_source_volume(confidence_components.n_sources),
                    "credibility_score": round(confidence_components.c_credibility, 3),
                    "agreement_proportion": round(confidence_components.a_agreement, 3),
                    "hedging_density": round(confidence_components.h_hedging, 3),
                    "tone_factor": round(confidence_components.t_tone, 3),
                    "evidence_score": round(confidence_components.evidence_score, 3),
                    "tone_score": round(confidence_components.tone_score, 3)
                },
                "explanation": explanation,
                "verification": response,
                "method": method,
                "evidence_sources": evidence_sources,
                "source_details": [
                    {
                        "type": s.source_type,
                        "credibility": s.credibility_score,
                        "supports": s.supports_claim,
                        "title": s.title,
                        "url": s.url
                    } for s in sources
                ]
            }
            
        except Exception as e:
            print(f"Error verifying claim: {str(e)}")
            return {
                "claim": claim,
                "type": "Unknown",
                "verdict": "ERROR",
                "confidence": 0,
                "confidence_interpretation": "Error in verification",
                "confidence_components": {},
                "explanation": f"Error during verification: {str(e)}",
                "verification": f"Error: {str(e)}",
                "method": "error",
                "evidence_sources": [],
                "source_details": []
            }

def process_claims_parallel(claims: List[str], llm, max_workers: int = 3) -> List[Dict]:
    """Process multiple claims in parallel"""
    verifier = EnhancedClaimVerifier(llm)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_claim = {
            executor.submit(verifier.verify_claim, claim): claim 
            for claim in claims
        }
        
        results = []
        for future in concurrent.futures.as_completed(future_to_claim):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                claim = future_to_claim[future]
                print(f"Error processing claim '{claim[:50]}...': {str(e)}")
                results.append({
                    "claim": claim,
                    "verdict": "ERROR",
                    "confidence": 0,
                    "explanation": f"Processing error: {str(e)}"
                })
    
    return results

def read_claims(file_path: str) -> List[str]:
    """Read claims from a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            return []
        
        # Split by lines and filter out empty lines
        claims = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Filter out lines that don't look like claims
        filtered_claims = []
        for claim in claims:
            if len(claim) > 10 and not claim.startswith('#'):
                filtered_claims.append(claim)
        
        return filtered_claims
        
    except Exception as e:
        print(f"Error reading claims from {file_path}: {str(e)}")
        return []

def main():
    """Main function with hardcoded test claims"""
    try:
        start_time = time.time()
        print("Starting enhanced claim verification process...")
        
        # Hardcoded test claims for demonstration
        test_claims = [
            "The Earth is approximately 4.5 billion years old according to scientific research.",
            "Drinking 8 glasses of water per day is necessary for optimal health.",
            "Albert Einstein said 'Imagination is more important than knowledge.'",
            "Climate change is caused primarily by human activities.",
            "Vaccines contain microchips for government tracking.",
            "The Great Wall of China is visible from space with the naked eye.",
            "Eating carrots improves night vision significantly.",
            "Coffee consumption increases the risk of heart disease.",
            "The human brain uses only 10% of its capacity.",
            "Goldfish have a 3-second memory span."
        ]
        
        print(f"Processing {len(test_claims)} hardcoded test claims...")
        
        # Initialize the LLM
        print("Initializing LLM...")
        if not OPENAI_API_KEY:
            print("ERROR: OpenAI API key not found. Using mock responses.")
            # In real implementation, you'd handle this differently
            sys.exit(1)
            
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Using a reliable model
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Create results directory
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Process claims
        print("Processing claims with enhanced confidence scoring...")
        verification_results = process_claims_parallel(test_claims, llm, max_workers=2)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Save results
        print("Saving verification results...")
        json_path = os.path.join(results_dir, "enhanced_verification_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(verification_results, f, indent=2, ensure_ascii=False)
        
        # Generate comprehensive report
        print("Generating enhanced verification report...")
        report_path = os.path.join(results_dir, "enhanced_verification_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Enhanced Claim Verification Report\n\n")
            f.write("## Mathematical Confidence Scoring System\n\n")
            f.write("This report uses a rigorous mathematical approach to calculate confidence scores based on:\n")
            f.write("- **N**: Number of distinct evidence sources\n")
            f.write("- **C**: Mean credibility score of sources (0-1)\n") 
            f.write("- **A**: Agreement proportion of sources (0-1)\n")
            f.write("- **H**: Hedging word density in LLM response (0-1)\n")
            f.write("- **T**: Tone factor (1-H)\n\n")
            f.write("**Formula**: Final Confidence = (0.7 × Evidence Score) + (0.3 × Tone Score)\n")
            f.write("Where Evidence Score = (0.2 × N) + (0.5 × C) + (0.3 × A)\n\n")
            
            for result in verification_results:
                f.write(f"## Claim\n{result['claim']}\n\n")
                f.write(f"**Type:** {result['type']}\n\n")
                f.write(f"**Verdict:** {result['verdict']}\n\n")
                f.write(f"**Confidence:** {result['confidence']}% ({result.get('confidence_interpretation', 'Unknown')})\n\n")
                
                # Add detailed confidence breakdown
                if 'confidence_components' in result:
                    comp = result['confidence_components']
                    f.write("**Confidence Breakdown:**\n")
                    f.write(f"- Sources Found: {comp.get('n_sources', 0)}\n")
                    f.write(f"- Source Volume Score: {comp.get('n_mapped', 0):.3f}\n")
                    f.write(f"- Average Credibility: {comp.get('credibility_score', 0):.3f}\n")
                    f.write(f"- Agreement Proportion: {comp.get('agreement_proportion', 0):.3f}\n")
                    f.write(f"- Hedging Density: {comp.get('hedging_density', 0):.3f}\n")
                    f.write(f"- Tone Factor: {comp.get('tone_factor', 0):.3f}\n")
                    f.write(f"- Evidence Score: {comp.get('evidence_score', 0):.3f}\n")
                    f.write(f"- Final Score: {comp.get('evidence_score', 0) * 0.7 + comp.get('tone_factor', 0) * 0.3:.3f}\n\n")
                
                f.write(f"**Evidence Sources:** {len(result.get('source_details', []))}\n\n")
                
                # Show source details
                if result.get('source_details'):
                    f.write("**Source Details:**\n")
                    for i, source in enumerate(result['source_details'], 1):
                        support_text = {True: "✓ Supports", False: "✗ Opposes", None: "○ Neutral"}.get(source['supports'], "? Unknown")
                        f.write(f"{i}. {source['type'].title()} (Credibility: {source['credibility']:.2f}) - {support_text}\n")
                        if source.get('title'):
                            f.write(f"   Title: {source['title']}\n")
                    f.write("\n")
                
                f.write(f"**Explanation:**\n{result['explanation']}\n\n")
                f.write("---\n\n")
        
        # Generate summary with mathematical analysis
        print("Generating enhanced summary...")
        summary_path = os.path.join(results_dir, "enhanced_verification_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Enhanced Claim Verification Summary\n\n")
            
            total_claims = len(verification_results)
            f.write(f"**Total Claims Processed:** {total_claims}\n")
            f.write(f"**Processing Time:** {elapsed_time:.2f} seconds\n")
            f.write(f"**Average Time per Claim:** {elapsed_time/total_claims:.2f} seconds\n\n")
            
            # Verdict distribution
            verdict_counts = {"TRUE": 0, "FALSE": 0, "CAN'T SAY": 0, "ERROR": 0}
            confidence_sum = {"TRUE": 0, "FALSE": 0, "CAN'T SAY": 0}
            confidence_count = {"TRUE": 0, "FALSE": 0, "CAN'T SAY": 0}
            
            for result in verification_results:
                verdict = result.get("verdict", "ERROR")
                confidence = result.get("confidence", 0)
                
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                
                if verdict in confidence_sum and confidence > 0:
                    confidence_sum[verdict] += confidence
                    confidence_count[verdict] += 1
            
            f.write("## Verdict Distribution\n\n")
            for verdict, count in verdict_counts.items():
                if count > 0:
                    percentage = (count / total_claims) * 100
                    avg_confidence = confidence_sum.get(verdict, 0) / confidence_count.get(verdict, 1) if confidence_count.get(verdict, 0) > 0 else 0
                    f.write(f"- **{verdict}**: {count} claims ({percentage:.1f}%) - Avg Confidence: {avg_confidence:.1f}%\n")
            
            # Confidence distribution analysis
            f.write("\n## Confidence Score Analysis\n\n")
            
            # Group by confidence ranges
            confidence_ranges = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
            range_counts = {range_tuple: 0 for range_tuple in confidence_ranges}
            
            valid_confidences = [r.get('confidence', 0) for r in verification_results if r.get('confidence', 0) > 0]
            
            for confidence in valid_confidences:
                for min_conf, max_conf in confidence_ranges:
                    if min_conf <= confidence <= max_conf:
                        range_counts[(min_conf, max_conf)] += 1
                        break
            
            f.write("**Confidence Range Distribution:**\n")
            for (min_conf, max_conf), count in range_counts.items():
                if count > 0:
                    percentage = (count / len(valid_confidences)) * 100 if valid_confidences else 0
                    range_name = {
                        (0, 20): "Very Low (0-20%)",
                        (21, 40): "Low (21-40%)", 
                        (41, 60): "Moderate (41-60%)",
                        (61, 80): "High (61-80%)",
                        (81, 100): "Very High (81-100%)"
                    }[(min_conf, max_conf)]
                    
                    f.write(f"- {range_name}: {count} claims ({percentage:.1f}%)\n")
            
            if valid_confidences:
                avg_confidence = sum(valid_confidences) / len(valid_confidences)
                f.write(f"\n**Overall Average Confidence:** {avg_confidence:.1f}%\n")
            
            # Evidence analysis
            f.write("\n## Evidence Source Analysis\n\n")
            
            total_sources = sum(len(r.get('source_details', [])) for r in verification_results)
            claims_with_evidence = sum(1 for r in verification_results if r.get('source_details'))
            
            f.write(f"- **Total Evidence Sources Found:** {total_sources}\n")
            f.write(f"- **Claims with Evidence:** {claims_with_evidence}/{total_claims} ({(claims_with_evidence/total_claims)*100:.1f}%)\n")
            f.write(f"- **Average Sources per Claim:** {total_sources/total_claims:.1f}\n")
            
            if claims_with_evidence > 0:
                f.write(f"- **Average Sources per Claim with Evidence:** {total_sources/claims_with_evidence:.1f}\n")
            
            # Source type breakdown
            source_type_counts = {}
            for result in verification_results:
                for source in result.get('source_details', []):
                    source_type = source['type']
                    source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
            
            if source_type_counts:
                f.write("\n**Source Type Distribution:**\n")
                for source_type, count in source_type_counts.items():
                    percentage = (count / total_sources) * 100 if total_sources > 0 else 0
                    f.write(f"- {source_type.title()}: {count} sources ({percentage:.1f}%)\n")
            
            # Credibility analysis
            all_credibility_scores = []
            for result in verification_results:
                for source in result.get('source_details', []):
                    if source.get('credibility') is not None:
                        all_credibility_scores.append(source['credibility'])
            
            if all_credibility_scores:
                avg_credibility = sum(all_credibility_scores) / len(all_credibility_scores)
                f.write(f"\n**Average Source Credibility:** {avg_credibility:.3f}/1.0\n")
                
                # Credibility distribution
                high_cred = sum(1 for c in all_credibility_scores if c >= 0.8)
                med_cred = sum(1 for c in all_credibility_scores if 0.5 <= c < 0.8)
                low_cred = sum(1 for c in all_credibility_scores if c < 0.5)
                
                f.write("\n**Source Credibility Distribution:**\n")
                f.write(f"- High Credibility (≥0.8): {high_cred} sources ({(high_cred/len(all_credibility_scores))*100:.1f}%)\n")
                f.write(f"- Medium Credibility (0.5-0.79): {med_cred} sources ({(med_cred/len(all_credibility_scores))*100:.1f}%)\n")
                f.write(f"- Low Credibility (<0.5): {low_cred} sources ({(low_cred/len(all_credibility_scores))*100:.1f}%)\n")
            
            # Agreement analysis
            supporting_sources = sum(1 for r in verification_results for s in r.get('source_details', []) if s.get('supports') is True)
            opposing_sources = sum(1 for r in verification_results for s in r.get('source_details', []) if s.get('supports') is False)
            neutral_sources = sum(1 for r in verification_results for s in r.get('source_details', []) if s.get('supports') is None)
            
            if total_sources > 0:
                f.write("\n## Source Agreement Analysis\n\n")
                f.write(f"- **Supporting Sources:** {supporting_sources} ({(supporting_sources/total_sources)*100:.1f}%)\n")
                f.write(f"- **Opposing Sources:** {opposing_sources} ({(opposing_sources/total_sources)*100:.1f}%)\n")
                f.write(f"- **Neutral/Unclear Sources:** {neutral_sources} ({(neutral_sources/total_sources)*100:.1f}%)\n")
            
            # Mathematical scoring insights
            f.write("\n## Mathematical Scoring Insights\n\n")
            
            # Calculate component averages
            n_components = []
            c_components = []
            a_components = []
            h_components = []
            evidence_scores = []
            tone_scores = []
            
            for result in verification_results:
                comp = result.get('confidence_components', {})
                if comp:
                    n_components.append(comp.get('n_sources', 0))
                    c_components.append(comp.get('credibility_score', 0))
                    a_components.append(comp.get('agreement_proportion', 0.5))
                    h_components.append(comp.get('hedging_density', 0))
                    evidence_scores.append(comp.get('evidence_score', 0))
                    tone_scores.append(comp.get('tone_score', 1))
            
            if n_components:
                f.write("**Average Component Scores:**\n")
                f.write(f"- Sources per Claim (N): {sum(n_components)/len(n_components):.1f}\n")
                f.write(f"- Credibility Score (C): {sum(c_components)/len(c_components):.3f}\n")
                f.write(f"- Agreement Proportion (A): {sum(a_components)/len(a_components):.3f}\n")
                f.write(f"- Hedging Density (H): {sum(h_components)/len(h_components):.3f}\n")
                f.write(f"- Evidence Score: {sum(evidence_scores)/len(evidence_scores):.3f}\n")
                f.write(f"- Tone Score: {sum(tone_scores)/len(tone_scores):.3f}\n")
            
            # Quality indicators
            f.write("\n## Quality Indicators\n\n")
            
            high_confidence_claims = sum(1 for r in verification_results if r.get('confidence', 0) >= 80)
            low_confidence_claims = sum(1 for r in verification_results if r.get('confidence', 0) <= 40)
            well_supported_claims = sum(1 for r in verification_results if len(r.get('source_details', [])) >= 3)
            
            f.write(f"- **High Confidence Claims (≥80%):** {high_confidence_claims}/{total_claims} ({(high_confidence_claims/total_claims)*100:.1f}%)\n")
            f.write(f"- **Low Confidence Claims (≤40%):** {low_confidence_claims}/{total_claims} ({(low_confidence_claims/total_claims)*100:.1f}%)\n")
            f.write(f"- **Well-Supported Claims (≥3 sources):** {well_supported_claims}/{total_claims} ({(well_supported_claims/total_claims)*100:.1f}%)\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            
            if low_confidence_claims > total_claims * 0.3:
                f.write(" **High proportion of low-confidence claims detected.** Consider:\n")
                f.write("- Expanding evidence gathering capabilities\n")
                f.write("- Improving source credibility assessment\n")
                f.write("- Enhancing claim-specific search strategies\n\n")
            
            if claims_with_evidence < total_claims * 0.7:
                f.write(" **Many claims lack external evidence.** Consider:\n")
                f.write("- Broadening search scope and databases\n")
                f.write("- Implementing specialized search for different claim types\n")
                f.write("- Adding more source types (patents, expert opinions, etc.)\n\n")
            
            if all_credibility_scores and sum(all_credibility_scores)/len(all_credibility_scores) < 0.6:
                f.write(" **Average source credibility is relatively low.** Consider:\n")
                f.write("- Prioritizing high-credibility sources in search\n")
                f.write("- Implementing more sophisticated credibility assessment\n")
                f.write("- Adding domain-specific credibility factors\n\n")
            
            f.write("\n## Technical Details\n\n")
            f.write("**Confidence Calculation Formula:**\n")
            f.write("```\n")
            f.write("N_mapped = log(N + 1) / log(N_max + 1)  # Source volume with diminishing returns\n")
            f.write("Evidence_Score = 0.2 × N_mapped + 0.5 × C + 0.3 × A\n")
            f.write("Tone_Score = 1 - H  # Where H is hedging density\n")
            f.write("Final_Confidence = 0.7 × Evidence_Score + 0.3 × Tone_Score\n")
            f.write("```\n\n")
            
            f.write("**Hedging Words Detected:**\n")
            hedging_words = EnhancedConfidenceCalculator.HEDGING_WORDS
            f.write(f"```\n{', '.join(hedging_words)}\n```\n\n")
            
            f.write("**Confidence Interpretation Ranges:**\n")
            for (min_conf, max_conf), interpretation in EnhancedConfidenceCalculator.CONFIDENCE_BOUNDARIES.items():
                f.write(f"- {min_conf}-{max_conf}%: {interpretation}\n")
        
        print(f"\nEnhanced claim verification completed in {elapsed_time:.2f} seconds!")
        print(f"Results saved to:")
        print(f"  - JSON: {json_path}")
        print(f"  - Report: {report_path}")
        print(f"  - Summary: {summary_path}")
        
        # Print quick summary to console
        print(f"\nQuick Summary:")
        print(f"- Total claims: {total_claims}")
        print(f"- Average confidence: {sum(r.get('confidence', 0) for r in verification_results)/len(verification_results):.1f}%")
        print(f"- Claims with evidence: {sum(1 for r in verification_results if r.get('source_details'))}")
        print(f"- Verdict distribution: {dict(verdict_counts)}")
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
