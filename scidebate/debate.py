"""Debate orchestrator.

Phase 2 features:
    - Heter-MAD: Pro/Con/Judge dùng LLM khác nhau
    - Uncertainty metrics: entropy + JSD + calibrated confidence
    - Parallel opening: Pro và Con opening chạy đồng thời
"""
import asyncio
import time
from dataclasses import dataclass, field
from scidebate.llms import BaseLLM
from scidebate.agents import ProAgent, ConAgent, JudgeAgent, Verdict
from scidebate.prompts import (
    pro_opening_prompt,
    con_opening_prompt,
    con_independent_opening_prompt,
    pro_rebuttal_prompt,
    con_rebuttal_prompt,
    judge_verdict_prompt,
    classify_claim,
)
from scidebate.uncertainty import compute_consensus, compute_disagreement, ConsensusMetrics
from scidebate.filtering import apply_dar_filtering
from scidebate.tools import retrieve_evidence


@dataclass
class Turn:
    """Một lượt phát biểu trong debate."""
    round_num: int
    speaker: str
    content: str
    model: str = ""
    filtered_out: bool = False
    filter_reason: str = ""

    def __str__(self):
        model_info = f" [{self.model}]" if self.model else ""
        filter_info = " [DAR FILTERED]" if self.filtered_out else ""
        return f"[Round {self.round_num}] {self.speaker}{model_info}{filter_info}:\n{self.content}"


@dataclass
class DebateResult:
    """Kết quả của một cuộc debate."""
    claim: str
    transcript: list[Turn] = field(default_factory=list)
    verdict: Verdict = None
    num_rounds: int = 0
    models_used: dict = field(default_factory=dict)
    consensus: ConsensusMetrics = None
    parallel_opening_used: bool = False
    elapsed_seconds: float = 0.0
    retrieved_papers: list[dict] = field(default_factory=list)
    pro_papers: list[dict] = field(default_factory=list)
    con_papers: list[dict] = field(default_factory=list)

    def to_text(self) -> str:
        return "\n\n".join(str(turn) for turn in self.transcript)

    def summary(self) -> str:
        if not self.verdict:
            return f"Claim: {self.claim}\nNo verdict yet."

        models_info = ""
        if self.models_used:
            models_info = "\nModels: " + " | ".join(
                f"{role}={model}" for role, model in self.models_used.items()
            )

        parallel_info = ""
        if self.parallel_opening_used:
            parallel_info = "\nParallel opening: enabled"

        time_info = ""
        if self.elapsed_seconds > 0:
            time_info = f"\nTime: {self.elapsed_seconds:.1f}s"

        consensus_info = ""
        if self.consensus:
            consensus_info = f"\n--- Consensus ---\n{self.consensus.summary()}"

        return (
            f"Claim: {self.claim}\n"
            f"Verdict: {self.verdict.verdict} "
            f"(confidence={self.verdict.confidence:.2f})\n"
            f"Justification: {self.verdict.justification}\n"
            f"Rounds: {self.num_rounds}"
            f"{models_info}"
            f"{parallel_info}"
            f"{time_info}"
            f"{consensus_info}"
        )


class Debate:
    """Multi-Agent Debate orchestrator.

    Usage:
        # Sequential (Phase 1 compat)
        debate = Debate(llm=OllamaLLM(...))
        result = debate.run("claim")

        # Parallel opening + uncertainty
        debate = Debate(
            pro_llm=..., con_llm=..., judge_llm=...,
            parallel_opening=True,
            compute_uncertainty=True,
        )
        result = debate.run("claim")
    """

    def __init__(
        self,
        llm: BaseLLM = None,
        pro_llm: BaseLLM = None,
        con_llm: BaseLLM = None,
        judge_llm: BaseLLM = None,
        max_rounds: int = 2,
        parallel_opening: bool = False,
        compute_uncertainty: bool = False,
        n_uncertainty_samples: int = 5,
        verbose: bool = True,
        uncertainty_pro_llm: BaseLLM = None,
        uncertainty_con_llm: BaseLLM = None,
        uncertainty_judge_llm: BaseLLM = None,
        on_turn_complete: callable = None,
        use_dar: bool = False,
        filter_llm: BaseLLM = None,
        enable_early_stopping: bool = False,
        use_logprobs: bool = True,
        on_consensus_complete: callable = None,
        enable_rag: bool = False,
        rag_max_results: int = 3,
        rag_source: str = "arxiv",
    ):
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

        self.pro_llm = pro_llm or llm
        self.con_llm = con_llm or llm
        self.judge_llm = judge_llm or llm

        if not self.pro_llm:
            raise ValueError("No LLM for Pro. Pass llm= or pro_llm=")
        if not self.con_llm:
            raise ValueError("No LLM for Con. Pass llm= or con_llm=")
        if not self.judge_llm:
            raise ValueError("No LLM for Judge. Pass llm= or judge_llm=")

        self.max_rounds = max_rounds
        self.parallel_opening = parallel_opening
        self.compute_uncertainty = compute_uncertainty
        self.n_uncertainty_samples = n_uncertainty_samples
        self.verbose = verbose
        self.on_turn_complete = on_turn_complete
        # LLM riêng cho uncertainty sampling (optional, mặc định dùng pro/con/judge_llm)
        self.uncertainty_pro_llm = uncertainty_pro_llm or self.pro_llm
        self.uncertainty_con_llm = uncertainty_con_llm or self.con_llm
        self.uncertainty_judge_llm = uncertainty_judge_llm or self.judge_llm
        self.use_dar = use_dar
        self.filter_llm = filter_llm or self.judge_llm
        self.enable_early_stopping = enable_early_stopping
        self.use_logprobs = use_logprobs
        self.on_consensus_complete = on_consensus_complete
        self.enable_rag = enable_rag
        self.rag_max_results = rag_max_results
        self.rag_source = rag_source

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _is_heterogeneous(self) -> bool:
        models = {self.pro_llm.model, self.con_llm.model, self.judge_llm.model}
        return len(models) > 1

    def run(
        self,
        claim: str,
        pre_retrieved_res: dict = None,
        pre_retrieved_pro: dict = None,
        pre_retrieved_con: dict = None
    ) -> DebateResult:
        """Chạy debate. Tự chọn sync hoặc async dựa trên parallel_opening."""
        # Classify claim type for universal claim handling
        self._claim_info = classify_claim(claim)
        self._claim_type = self._claim_info["claim_type"]

        pro_context = ""
        con_context = ""
        pro_papers = []
        con_papers = []

        if self.enable_rag:
            if pre_retrieved_pro and pre_retrieved_con:
                pro_context = pre_retrieved_pro.get("context", "")
                pro_papers = pre_retrieved_pro.get("papers", [])
                con_context = pre_retrieved_con.get("context", "")
                con_papers = pre_retrieved_con.get("papers", [])
            elif pre_retrieved_res:
                # Backward compatibility fallback
                pro_context = pre_retrieved_res.get("context", "")
                pro_papers = pre_retrieved_res.get("papers", [])
                con_context = pre_retrieved_res.get("context", "")
                con_papers = pre_retrieved_res.get("papers", [])
            else:
                self._log(f"\nRetrieving supporting evidence (PRO) from {self.rag_source} for: '{claim}'...")
                pro_res = retrieve_evidence(
                    claim,
                    max_results=self.rag_max_results,
                    generator_llm=self.pro_llm,
                    stance="PRO",
                    source=self.rag_source,
                )
                pro_context = pro_res["context"]
                pro_papers = pro_res["papers"]

                self._log(f"\nRetrieving opposing evidence (CON) from {self.rag_source} for: '{claim}'...")
                con_res = retrieve_evidence(
                    claim,
                    max_results=self.rag_max_results,
                    generator_llm=self.con_llm,
                    stance="CON",
                    source=self.rag_source,
                )
                con_context = con_res["context"]
                con_papers = con_res["papers"]

                self._log(f"Retrieved {len(pro_papers)} PRO papers and {len(con_papers)} CON papers.")

        if self.parallel_opening:
            # asyncio.run() tạo event loop mới
            return asyncio.run(self._run_async(claim, pro_context, con_context, pro_papers, con_papers))
        else:
            return self._run_sync(claim, pro_context, con_context, pro_papers, con_papers)

    def _run_sync(
        self,
        claim: str,
        pro_context: str = "",
        con_context: str = "",
        pro_papers: list = None,
        con_papers: list = None
    ) -> DebateResult:
        """Chạy debate tuần tự (Phase 1 behavior)."""
        start_time = time.time()

        pro = ProAgent(llm=self.pro_llm)
        con = ConAgent(llm=self.con_llm)
        judge = JudgeAgent(llm=self.judge_llm)

        result = DebateResult(
            claim=claim,
            models_used={
                "PRO": self.pro_llm.model,
                "CON": self.con_llm.model,
                "JUDGE": self.judge_llm.model,
            },
            parallel_opening_used=False,
            retrieved_papers=(pro_papers or []) + (con_papers or []),
            pro_papers=pro_papers or [],
            con_papers=con_papers or [],
        )

        heter_tag = " [HETER-MAD]" if self._is_heterogeneous() else ""
        self._log(f"\n{'=' * 60}")
        self._log(f"DEBATE{heter_tag} [SEQUENTIAL]: {claim}")
        self._log(f"  PRO   → {self.pro_llm.model}")
        self._log(f"  CON   → {self.con_llm.model}")
        self._log(f"  JUDGE → {self.judge_llm.model}")
        self._log(f"{'=' * 60}")

        # Round 1: sequential opening
        self._log("\n----- Round 1: Opening (sequential) -----")
        _ct = getattr(self, '_claim_type', 'GENERAL')

        pro_prompt = pro_opening_prompt(claim, claim_type=_ct)
        if pro_context:
            pro_prompt = f"{pro_context}\n\nBased on the scientific evidence base above, address the claim:\n{pro_prompt}"
        pro_opening = pro.respond(pro_prompt)

        pro_turn = Turn(1, "PRO", pro_opening, self.pro_llm.model)
        if self.use_dar:
            pro_turn.filter_reason = "Lượt mở đầu - tự động giữ lại."
        result.transcript.append(pro_turn)
        if self.on_turn_complete:
            self.on_turn_complete(pro_turn)
        self._log(f"\n[PRO | {self.pro_llm.model}]\n{pro_opening}\n")

        con_prompt = con_opening_prompt(claim, pro_opening, claim_type=_ct)
        if con_context:
            con_prompt = f"{con_context}\n\nBased on the scientific evidence base above, address the claim:\n{con_prompt}"
        con_opening = con.respond(con_prompt)
        
        con_turn = Turn(1, "CON", con_opening, self.con_llm.model)
        if self.use_dar:
            con_turn.filter_reason = "Lượt mở đầu - tự động giữ lại."
        result.transcript.append(con_turn)
        if self.on_turn_complete:
            self.on_turn_complete(con_turn)
        self._log(f"\n[CON | {self.con_llm.model}]\n{con_opening}\n")

        retained_turns = [pro_turn, con_turn]

        # Rebuttals + Judge + Uncertainty
        self._run_rebuttals_judge_uncertainty(
            pro, con, judge, claim, pro_opening, con_opening, result, retained_turns, pro_context, con_context
        )

        result.elapsed_seconds = time.time() - start_time
        return result

    async def _run_async(
        self,
        claim: str,
        pro_context: str = "",
        con_context: str = "",
        pro_papers: list = None,
        con_papers: list = None
    ) -> DebateResult:
        """Chạy debate với parallel opening."""
        start_time = time.time()

        pro = ProAgent(llm=self.pro_llm)
        con = ConAgent(llm=self.con_llm)
        judge = JudgeAgent(llm=self.judge_llm)

        result = DebateResult(
            claim=claim,
            models_used={
                "PRO": self.pro_llm.model,
                "CON": self.con_llm.model,
                "JUDGE": self.judge_llm.model,
            },
            parallel_opening_used=True,
            retrieved_papers=(pro_papers or []) + (con_papers or []),
            pro_papers=pro_papers or [],
            con_papers=con_papers or [],
        )

        heter_tag = " [HETER-MAD]" if self._is_heterogeneous() else ""
        self._log(f"\n{'=' * 60}")
        self._log(f"DEBATE{heter_tag} [PARALLEL OPENING]: {claim}")
        self._log(f"  PRO   → {self.pro_llm.model}")
        self._log(f"  CON   → {self.con_llm.model}")
        self._log(f"  JUDGE → {self.judge_llm.model}")
        self._log(f"{'=' * 60}")

        # Round 1: parallel opening
        self._log("\n----- Round 1: Opening (parallel) -----")
        _ct = getattr(self, '_claim_type', 'GENERAL')

        async def run_pro():
            pro_prompt = pro_opening_prompt(claim, claim_type=_ct)
            if pro_context:
                pro_prompt = f"{pro_context}\n\nBased on the scientific evidence base above, address the claim:\n{pro_prompt}"
            return await pro.respond_async(pro_prompt)

        async def run_con():
            con_prompt = con_independent_opening_prompt(claim, claim_type=_ct)
            if con_context:
                con_prompt = f"{con_context}\n\nBased on the scientific evidence base above, address the claim:\n{con_prompt}"
            return await con.respond_async(con_prompt)

        pro_text, con_text = await asyncio.gather(run_pro(), run_con())

        pro_turn = Turn(1, "PRO", pro_text, self.pro_llm.model)
        if self.use_dar:
            pro_turn.filter_reason = "Lượt mở đầu - tự động giữ lại."
        result.transcript.append(pro_turn)
        if self.on_turn_complete:
            self.on_turn_complete(pro_turn)
        self._log(f"\n[PRO | {self.pro_llm.model}]\n{pro_text}\n")

        con_turn = Turn(1, "CON", con_text, self.con_llm.model)
        if self.use_dar:
            con_turn.filter_reason = "Lượt mở đầu - tự động giữ lại."
        result.transcript.append(con_turn)
        if self.on_turn_complete:
            self.on_turn_complete(con_turn)
        self._log(f"\n[CON | {self.con_llm.model}]\n{con_text}\n")

        retained_turns = [pro_turn, con_turn]

        # Rebuttals + Judge + Uncertainty (tuần tự từ đây)
        self._run_rebuttals_judge_uncertainty(
            pro, con, judge, claim, pro_turn.content, con_turn.content, result, retained_turns, pro_context, con_context
        )

        result.elapsed_seconds = time.time() - start_time
        return result

    def _run_rebuttals_judge_uncertainty(
        self,
        pro: ProAgent,
        con: ConAgent,
        judge: JudgeAgent,
        claim: str,
        last_pro: str,
        last_con: str,
        result: DebateResult,
        retained_turns: list,
        pro_context: str = "",
        con_context: str = "",
    ) -> None:
        """Phần chung: rebuttals + judge + uncertainty. Dùng cho cả sync và async."""

        # Khởi tạo feedback rỗng
        pro_jsd_feedback = ""
        con_jsd_feedback = ""
        _ct = getattr(self, '_claim_type', 'GENERAL')

        # Rebuttals
        for round_num in range(2, self.max_rounds + 1):
            self._log(f"\n----- Round {round_num}: Rebuttals -----")

            # 1. PRO rebuttal
            pro_prompt = pro_rebuttal_prompt(claim, last_con, feedback=pro_jsd_feedback, claim_type=_ct)
            if pro_context:
                pro_prompt = f"{pro_context}\n\nBased on the scientific evidence base above, write your rebuttal:\n{pro_prompt}"
            pro_rebut = pro.respond(pro_prompt)
            pro_turn = Turn(round_num, "PRO", pro_rebut, self.pro_llm.model)
            
            if self.use_dar:
                res = apply_dar_filtering(claim, retained_turns, pro_turn, self.filter_llm)
                pro_turn.filter_reason = res["reason"]
                if res["decision"] == "DROP":
                    pro_turn.filtered_out = True
                    pro.memory.remove_last(2)  # Xóa prompt và response bị lặp
                    self._log(f"\n[DAR FILTERED - PRO ROUND {round_num} DROPPED]: {res['reason']}\n")
                else:
                    retained_turns.append(pro_turn)
                    last_pro = pro_rebut
            else:
                retained_turns.append(pro_turn)
                last_pro = pro_rebut

            result.transcript.append(pro_turn)
            if self.on_turn_complete:
                self.on_turn_complete(pro_turn)
            self._log(f"\n[PRO | {self.pro_llm.model}]\n{pro_rebut}\n")

            # 2. CON rebuttal
            con_prompt = con_rebuttal_prompt(claim, last_pro, feedback=con_jsd_feedback, claim_type=_ct)
            if con_context:
                con_prompt = f"{con_context}\n\nBased on the scientific evidence base above, write your rebuttal:\n{con_prompt}"
            con_rebut = con.respond(con_prompt)
            con_turn = Turn(round_num, "CON", con_rebut, self.con_llm.model)
            
            if self.use_dar:
                res = apply_dar_filtering(claim, retained_turns, con_turn, self.filter_llm)
                con_turn.filter_reason = res["reason"]
                if res["decision"] == "DROP":
                    con_turn.filtered_out = True
                    con.memory.remove_last(2)  # Xóa prompt và response bị lặp
                    self._log(f"\n[DAR FILTERED - CON ROUND {round_num} DROPPED]: {res['reason']}\n")
                else:
                    retained_turns.append(con_turn)
                    last_con = con_rebut
            else:
                retained_turns.append(con_turn)
                last_con = con_rebut

            result.transcript.append(con_turn)
            if self.on_turn_complete:
                self.on_turn_complete(con_turn)
            self._log(f"\n[CON | {self.con_llm.model}]\n{con_rebut}\n")

            # 3. Intermediate Disagreement & Early Stopping & Feedback
            jsd_feedback = ""
            if self.compute_uncertainty:
                # BƯỚC 1: Chỉ tính JSD (Pro/Con) trước để tối ưu hóa hiệu năng
                self._log(f"\nEvaluating Pro/Con Disagreement (JSD) at the end of Round {round_num}...")
                current_transcript_text = "\n\n".join(str(turn) for turn in retained_turns)
                
                disagreement_result = compute_disagreement(
                    claim=claim,
                    transcript_text=current_transcript_text,
                    pro_llm=self.uncertainty_pro_llm,
                    con_llm=self.uncertainty_con_llm,
                    n_samples=self.n_uncertainty_samples,
                    temperature=0.7,
                    use_logprobs=self.use_logprobs,
                )
                current_jsd = disagreement_result.jsd
                
                # Flag kiểm tra xem có thực sự đủ điều kiện dừng sớm (Q1) hay không
                should_early_stop = False
                
                # BƯỚC 2: TỐI ƯU HÓA CỦA BẠN
                # Chỉ khi JSD < 0.3 (Pro/Con đã đồng thuận) mới tốn chi phí tính Entropy của Judge
                if current_jsd < 0.3:
                    self._log(f"JSD is low ({current_jsd:.3f}). Verifying Judge stability (Entropy) to confirm Q1...")
                    
                    # Lấy phán quyết Judge nháp tại vòng này để lấy thông số confidence gốc
                    judge_prompt = judge_verdict_prompt(claim, current_transcript_text)
                    judge_raw = judge.respond(judge_prompt)
                    temp_verdict = judge.parse_verdict(judge_raw)
                    
                    # Tính toán đầy đủ Consensus (bao gồm cả Entropy của Judge)
                    consensus_round = compute_consensus(
                        claim=claim,
                        transcript_text=current_transcript_text,
                        pro_llm=self.uncertainty_pro_llm,
                        con_llm=self.uncertainty_con_llm,
                        judge_llm=self.judge_llm,
                        judge_raw_confidence=temp_verdict.confidence,
                        n_samples=self.n_uncertainty_samples,
                        uncertainty_judge_llm=self.uncertainty_judge_llm,
                        use_logprobs=self.use_logprobs,
                    )
                    
                    # Kiểm tra xem có thực sự là Q1 (Strong Consensus) hay không
                    if consensus_round.consensus_quadrant == "Strong Consensus":
                        should_early_stop = True
                        result.consensus = consensus_round  # Lưu lại consensus hoàn chỉnh để vẽ biểu đồ
                        self._log(f"\n[EARLY STOPPING CONFIRMED]: Strong Consensus (Q1) verified. Ending debate.")
                    else:
                        # Rơi vào Q3 (Aligned Uncertainty): Pro/Con đồng ý nhưng Judge chưa ổn định
                        self._log(
                            f"[EARLY STOPPING BYPASSED]: Pro/Con aligned but Judge is unstable "
                            f"(Quadrant={consensus_round.consensus_quadrant}, Entropy={consensus_round.normalized_entropy:.2f}). "
                            f"Continuing debate..."
                        )
                else:
                    # Nếu JSD >= 0.3, tạo phản hồi yêu cầu hai Agent tập trung làm rõ bất đồng
                    jsd_feedback = (
                        f"Note: There is currently a high level of disagreement "
                        f"(JSD = {current_jsd:.2f}) on the claim. Focus your next "
                        f"argument on addressing the core contradictions and providing concrete evidence."
                    )
                
                # BƯỚC 3: Cập nhật kết quả lên Streamlit UI dựa trên trạng thái dừng
                if self.on_consensus_complete:
                    if should_early_stop:
                        # Gửi Consensus Metrics đầy đủ để vẽ biểu đồ
                        self.on_consensus_complete(round_num, result.consensus)
                    else:
                        # Chỉ gửi thông tin JSD để vẽ/hiển thị trạng thái phân kỳ tạm thời
                        self.on_consensus_complete(round_num, {
                            "jsd": current_jsd,
                            "pro_dist": disagreement_result.pro_distribution.distribution,
                            "con_dist": disagreement_result.con_distribution.distribution
                        })
                
                # BƯỚC 4: Kích hoạt dừng sớm nếu thỏa mãn điều kiện
                if self.enable_early_stopping and should_early_stop:
                    result.num_rounds = round_num
                    result._early_stopped = True
                    break
            # pro_jsd_feedback = ""
            # con_jsd_feedback = ""
            # if self.compute_uncertainty:
            #     self._log(f"\nEvaluating Pro/Con Disagreement (JSD) at the end of Round {round_num}...")
            #     current_transcript_text = "\n\n".join(str(turn) for turn in retained_turns)

            #     disagreement_result = compute_disagreement(
            #         claim=claim,
            #         transcript_text=current_transcript_text,
            #         pro_llm=self.uncertainty_pro_llm,
            #         con_llm=self.uncertainty_con_llm,
            #         n_samples=self.n_uncertainty_samples,
            #         temperature=0.7,
            #         use_logprobs=self.use_logprobs,
            #         claim_type=_ct,
            #     )

            #     current_jsd = disagreement_result.jsd
            #     self._log(
            #         f"Round {round_num} Pro/Con Disagreement (JSD): {current_jsd:.3f} | "
            #         f"Pro: {disagreement_result.pro_distribution.distribution} | "
            #         f"Con: {disagreement_result.con_distribution.distribution}"
            #     )

            #     # Callback để Streamlit update UI
            #     if self.on_consensus_complete:
            #         self.on_consensus_complete(round_num, {
            #             "jsd": current_jsd,
            #             "pro_dist": disagreement_result.pro_distribution.distribution,
            #             "con_dist": disagreement_result.con_distribution.distribution,
            #         })

            #     # Role-specific feedback when disagreement is high
            #     if current_jsd > 0.3:
            #         pro_jsd_feedback = (
            #             f"ROUND FEEDBACK (JSD = {current_jsd:.2f} — agents strongly disagree): "
            #             f"CON is disputing your evidence as indirect or correlational. "
            #             f"If your evidence does not directly measure the causal mechanism, "
            #             f"acknowledge this limitation rather than continuing to assert SUPPORTED. "
            #             f"Direct causal evidence is required — proxy predictors are not sufficient."
            #         )
            #         con_jsd_feedback = (
            #             f"ROUND FEEDBACK (JSD = {current_jsd:.2f} — agents strongly disagree): "
            #             f"PRO is asserting the claim strongly despite weak evidence. "
            #             f"Reinforce the exact logical gap: name the specific error "
            #             f"(correlation vs. causation, proxy vs. direct, uncontrolled confounders). "
            #             f"Argue explicitly that without direct causal evidence, the verdict must be INCONCLUSIVE."
            #         )
            #         self._log(f"[FEEDBACK]: High JSD ({current_jsd:.2f}) — role-specific guidance generated.")

            #     # Early Stopping: If JSD is low (strong consensus), end debate early
            #     if self.enable_early_stopping and current_jsd < 0.3:
            #         self._log(
            #             f"\n[EARLY STOPPING TRIGGERED]: Low disagreement (JSD = {current_jsd:.3f}) detected. "
            #             f"Ending debate early at Round {round_num}."
            #         )
            #         result.num_rounds = round_num
            #         result._early_stopped = True
            #         break

        if result.num_rounds == 0:
            result.num_rounds = self.max_rounds

        # Judge verdict
        self._log("\n----- Final: Judge verdict -----")
        if self.use_dar:
            # Build transcript từ các turn được giữ lại
            transcript_text = "\n\n".join(str(turn) for turn in retained_turns)
        else:
            transcript_text = result.to_text()

        judge_prompt = judge_verdict_prompt(claim, transcript_text, claim_type=_ct)
        # Isolated Judge: DO NOT inject any RAG paper context.
        judge_raw = judge.respond(judge_prompt)
        verdict = judge.parse_verdict(judge_raw)
        result.verdict = verdict

        self._log(f"\n[JUDGE | {self.judge_llm.model}]\n{judge_raw}\n")
        self._log(
            f"\n{'=' * 60}\n"
            f"FINAL VERDICT: {verdict.verdict} "
            f"(confidence={verdict.confidence:.2f})\n"
            f"{'=' * 60}"
        )

        # Uncertainty: Compute FULL consensus ONLY at the end (after Judge verdict)
        # Wrapped in try/except so a 429 rate limit does NOT erase the already-computed verdict.
        if self.compute_uncertainty:
            self._log(
                f"\n----- Computing Final Consensus Metrics (Round {result.num_rounds}) -----\n"
                f"  Sampling {self.n_uncertainty_samples}× per agent..."
            )
            try:
                consensus = compute_consensus(
                    claim=claim,
                    transcript_text=transcript_text,
                    pro_llm=self.uncertainty_pro_llm,
                    con_llm=self.uncertainty_con_llm,
                    judge_llm=self.judge_llm,
                    judge_raw_confidence=verdict.confidence,
                    n_samples=self.n_uncertainty_samples,
                    uncertainty_judge_llm=self.uncertainty_judge_llm,
                    use_logprobs=self.use_logprobs,
                )
                result.consensus = consensus
                self._log(f"\n{result.consensus.summary()}")
                tag = "EARLY STOP" if getattr(result, '_early_stopped', False) else "FINAL"
                self._log(
                    f"\n{'=' * 60}\n"
                    f"{tag} CONSENSUS: {result.consensus.consensus_level} "
                    f"({result.consensus.consensus_quadrant})\n"
                    f"Entropy: {result.consensus.normalized_entropy:.2f} | "
                    f"JSD: {result.consensus.jsd:.3f} | "
                    f"Calibrated Confidence: {result.consensus.calibrated_confidence:.2f}\n"
                    f"{'=' * 60}"
                )
            except Exception as _consensus_err:
                result.consensus = None
                result._consensus_error = str(_consensus_err)
                self._log(f"\n[CONSENSUS SKIPPED — {_consensus_err}]")
