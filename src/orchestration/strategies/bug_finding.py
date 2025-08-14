"""
Bug Finding Strategy - Discovers and validates bugs in code.

This strategy runs two phases:
1. Discovery: Find potential bugs in specified code areas
2. Validation: Reproduce and document confirmed bugs with commits
"""

import logging
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

from ...shared import InstanceResult
from .base import Strategy, StrategyConfig

if TYPE_CHECKING:
    from ..strategy_context import StrategyContext

logger = logging.getLogger(__name__)


@dataclass
class BugFindingConfig(StrategyConfig):
    """Configuration for bug finding strategy."""
    
    # Target area to search for bugs (e.g., "src/parser", "authentication module")
    target_area: str = ""
    
    # Additional context about what kinds of bugs to look for
    bug_focus: str = ""
    
    # Prompt template for discovery phase
    discovery_prompt_template: str = """You are a security researcher and bug hunter. Your task is to find bugs in the {target_area} of this codebase.

{bug_focus_section}

Instructions:
1. Thoroughly analyze the code in the specified area
2. Look for any kind of bugs: logic errors, security vulnerabilities, race conditions, memory issues, edge cases, etc.
3. When you find a potential bug, try to reproduce it
4. Once you confirm a real bug, provide a detailed explanation including:
   - What the bug is
   - Why it happens (root cause analysis)
   - How to reproduce it step by step
   - Potential impact/severity
5. IMPORTANT: Stop after finding and documenting ONE confirmed bug

Focus area: {target_area}
Original request: {original_prompt}"""
    
    # Prompt template for validation phase  
    validation_prompt_template: str = """You are a senior engineer tasked with validating a bug report. Another researcher claims to have found the following bug:

{bug_report}

Your task:
1. Carefully read the bug report
2. Try to reproduce the bug following the provided steps
3. Analyze if this is a real issue or a false positive
4. If it's a VALID bug that you can reproduce:
   - Create a file named BUG_REPORT.md with a detailed bug report
   - Include: summary, severity, reproduction steps, root cause, affected components
   - Create any additional files needed for reproduction (test cases, PoC code, etc.)
   - IMPORTANT: Commit these files with a descriptive commit message
5. If you CANNOT reproduce it or it's not a real bug:
   - Do NOT create any files
   - Do NOT make any commits
   - Just explain why it's not valid"""

    def validate(self) -> None:
        """Validate configuration."""
        super().validate()
        if not self.target_area:
            raise ValueError("target_area must be specified for bug finding")


class BugFindingStrategy(Strategy):
    """
    Two-phase bug finding strategy.
    
    Phase 1: Discovery - Find and document a bug
    Phase 2: Validation - Reproduce and commit bug report
    """
    
    @property
    def name(self) -> str:
        return "bug-finding"
    
    def get_config_class(self) -> type[BugFindingConfig]:
        return BugFindingConfig
    
    async def execute(
        self,
        prompt: str,
        base_branch: str,
        ctx: "StrategyContext",
    ) -> List[InstanceResult]:
        """Execute bug finding strategy."""
        config: BugFindingConfig = self.create_config()  # type: ignore
        
        # Build discovery prompt
        bug_focus_section = ""
        if config.bug_focus:
            bug_focus_section = f"Specific focus: {config.bug_focus}\n"
            
        discovery_prompt = config.discovery_prompt_template.format(
            target_area=config.target_area,
            bug_focus_section=bug_focus_section,
            original_prompt=prompt
        )
        
        # Phase 1: Discovery
        logger.info(f"Starting bug discovery phase for {config.target_area}")
        # ctx.emit_event not available in current context
        logger.info(f"Bug finding phase: discovery for {config.target_area}")
        
        discovery_task = {"prompt": discovery_prompt, "base_branch": base_branch, "model": config.model}
        discovery_handle = await ctx.run(discovery_task, key=ctx.key("discovery", config.target_area))
        discovery_result = await ctx.wait(discovery_handle)
        
        # Check if discovery was successful
        if not discovery_result.success:
            logger.warning("Bug discovery phase failed")
            logger.warning(f"Discovery failed: {discovery_result.error}")
            return [discovery_result]
        
        # Extract bug report from discovery result
        bug_report = discovery_result.final_message or "No bug report provided"
        
        # Phase 2: Validation
        logger.info("Starting bug validation phase")
        logger.info(f"Bug finding phase: validation, bug_found={bool(bug_report)}")
        
        validation_prompt = config.validation_prompt_template.format(
            bug_report=bug_report
        )
        
        # Validation starts from the base branch, not discovery branch
        # This ensures independent reproduction
        validation_task = {"prompt": validation_prompt, "base_branch": base_branch, "model": config.model}
        validation_handle = await ctx.run(validation_task, key=ctx.key("validation", config.target_area))
        validation_result = await ctx.wait(validation_handle)
        
        # Determine final status based on validation
        if validation_result.success:
            # Check if validation instance made commits (indicating valid bug)
            commit_count = validation_result.commit_statistics.get("commit_count", 0) if validation_result.commit_statistics else 0
            if commit_count > 0:
                logger.info(f"Bug validated and documented with {commit_count} commits")
                logger.info(f"Bug confirmed in {config.target_area} with {commit_count} commits on branch {validation_result.branch_name}")
                
                # Mark validation result with success metadata
                validation_result.metadata["bug_confirmed"] = True
                validation_result.metadata["bug_report_branch"] = validation_result.branch_name
            else:
                logger.info("Bug could not be validated (no commits made)")
                logger.info("Bug not confirmed: No commits made during validation")
                
                # Override status to indicate validation failure
                validation_result.success = False
                validation_result.status = "failed"
                validation_result.error = "Bug could not be reproduced or was invalid"
                validation_result.metadata["bug_confirmed"] = False
        
        # Return both results for transparency
        return [discovery_result, validation_result]
