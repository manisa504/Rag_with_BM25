"""
Sample airline domain evaluation datasets.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QAExample:
    """Question-Answer example for evaluation."""
    question: str
    reference_answer: str
    category: str
    difficulty: str


class AirlineEvaluationDataset:
    """Sample airline domain Q&A dataset for evaluation."""
    
    def __init__(self):
        """Initialize with aircraft performance Q&A examples based on BADA corpus."""
        self.examples = [
            QAExample(
                question="What are the key differences between BADA 3 and BADA 4 aircraft performance models in terms of complex flight operations?",
                reference_answer="BADA 4 has significantly improved realism and accuracy compared to BADA 3, enabling support for complex flight operations like economy climb, cruise and descent based on cost index, maximum range cruise, and optimum altitude operations. While BADA 3 has limited capability for these complex operations due to insufficient accuracy and realism, BADA 4 can successfully model these advanced functions. The improved accuracy of BADA 4's underlying aircraft forces, including accurate estimation of drag, thrust and fuel consumption, supports modeling of complex operations that are not feasible with BADA 3.",
                category="aircraft_performance_models",
                difficulty="medium"
            ),
            QAExample(
                question="How does Cost Index (CI) affect aircraft economic cruise speed and what is the typical range of CI values?",
                reference_answer="Cost Index (CI) is defined as the ratio between time-related and fuel-related costs (CT/CF) measured in kg/min. The range of CI values typically varies from 0 to 99 or 999 kg/min depending on the aircraft manufacturer. A higher CI results in higher economic Mach number (MECON) - meaning faster cruise speeds to minimize time costs. CI = 0 represents minimum fuel mode (maximum fuel efficiency), while maximum CI represents minimum flight time mode. For example, CI = 20 means the cost of 20 kg of fuel equals the cost of 1 flight minute.",
                category="cost_optimization",
                difficulty="medium"
            ),
            QAExample(
                question="What is Maximum Range Cruise (MRC) and how does it differ from Long Range Cruise (LRC)?",
                reference_answer="Maximum Range Cruise (MRC) maximizes flight range for given fuel load and atmospheric conditions by achieving minimum fuel consumption per unit distance (maximum specific range). MRC finds the Mach number that provides maximum distance the aircraft can fly with given fuel. Long Range Cruise (LRC) provides 99% of the MRC specific range but allows slightly higher fuel consumption for significantly increased Mach number and reduced flight time. LRC is defined as the Mach number where specific range equals 99% of the maximum range specific range.",
                category="cruise_optimization",
                difficulty="hard"
            ),
            QAExample(
                question="What is the purpose of Maximum Endurance Cruise (MEC) and when is it typically used?",
                reference_answer="Maximum Endurance Cruise (MEC) is used for holding operations where the objective is to maximize the time an aircraft can remain airborne with a given amount of fuel. It minimizes fuel consumption with respect to time by finding the holding Mach number (Mmec) at minimum allowable fuel flow. This speed is typically near the minimum drag speed and maximum L/D speed, but is usually increased slightly to provide easier aircraft control since it falls in the speed-instability region.",
                category="holding_operations", 
                difficulty="medium"
            ),
            QAExample(
                question="How does aircraft weight affect optimum altitude and specific range during cruise?",
                reference_answer="As aircraft weight decreases during flight (due to fuel burn), both specific range and optimum altitude increase for a given Mach number. The optimum altitude corresponds to maximum lift-to-drag ratio (L/D) or maximum lift coefficient to drag coefficient ratio (CL/CD). During flight, as weight decreases, the aircraft should climb to higher altitudes to maintain optimum efficiency. Conversely, higher aircraft weight requires lower optimum altitudes for maximum efficiency.",
                category="flight_optimization",
                difficulty="hard"
            ),
            QAExample(
                question="What validation challenges exist when using BADA thrust models independently, particularly at different altitudes?",
                reference_answer="Independent validation of BADA thrust models faces several challenges: (1) Original engine data is not easily available for comparison; (2) At low altitudes (below 1500 ft), MTKF (Maximum Take-off) engine rating is not supported by BADA families, requiring use of MCMB (Maximum Climb) rating with constant multipliers; (3) From 1500-4000 ft, significant errors occur due to boundary behavior and limited reference data availability; (4) Above 4000 ft, both BADA 3 and 4 show acceptable accuracy, with BADA 4 demonstrating improved precision (0.5% vs 5.5% relative error).",
                category="model_validation",
                difficulty="hard"
            )
        ]
    
    def get_all_examples(self) -> List[QAExample]:
        """Get all evaluation examples."""
        return self.examples
    
    def get_examples_by_category(self, category: str) -> List[QAExample]:
        """Get examples filtered by category."""
        return [ex for ex in self.examples if ex.category == category]
    
    def get_examples_by_difficulty(self, difficulty: str) -> List[QAExample]:
        """Get examples filtered by difficulty."""
        return [ex for ex in self.examples if ex.difficulty == difficulty]
    
    def to_evaluation_format(self) -> List[Dict[str, Any]]:
        """Convert to format expected by evaluation pipeline."""
        return [
            {
                "question": ex.question,
                "reference_answer": ex.reference_answer,
                "category": ex.category,
                "difficulty": ex.difficulty
            }
            for ex in self.examples
        ]
    
    def get_categories(self) -> List[str]:
        """Get all unique categories in the dataset."""
        return list(set(ex.category for ex in self.examples))
    
    def get_difficulties(self) -> List[str]:
        """Get all unique difficulty levels in the dataset."""
        return list(set(ex.difficulty for ex in self.examples))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        categories = {}
        difficulties = {}
        
        for ex in self.examples:
            categories[ex.category] = categories.get(ex.category, 0) + 1
            difficulties[ex.difficulty] = difficulties.get(ex.difficulty, 0) + 1
        
        return {
            "total_examples": len(self.examples),
            "categories": categories,
            "difficulties": difficulties,
            "avg_question_length": sum(len(ex.question.split()) for ex in self.examples) / len(self.examples),
            "avg_answer_length": sum(len(ex.reference_answer.split()) for ex in self.examples) / len(self.examples)
        }


# Global dataset instance
airline_dataset = AirlineEvaluationDataset()


def get_evaluation_dataset() -> List[Dict[str, Any]]:
    """Get the evaluation dataset in the format expected by the pipeline."""
    return airline_dataset.to_evaluation_format()


def get_sample_questions(n: int = 5) -> List[Dict[str, Any]]:
    """Get a sample of n questions for quick testing."""
    examples = airline_dataset.get_all_examples()[:n]
    return [
        {
            "question": ex.question,
            "reference_answer": ex.reference_answer,
            "category": ex.category,
            "difficulty": ex.difficulty
        }
        for ex in examples
    ]
