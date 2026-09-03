from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RegulatorEnum(str, Enum):
    IEPFA = "IEPFA (Ministry of Corporate Affairs)"
    RBI_UDGAM = "RBI UDGAM (Reserve Bank of India)"
    SEBI = "SEBI / AMFI (Securities & Exchange Board of India)"
    IRDAI = "IRDAI Bima Bharosa (Insurance Regulatory & Dev Authority)"


class AssetTypeEnum(str, Enum):
    EQUITY_SHARES = "Equity Shares"
    UNCLAIMED_DIVIDEND = "Unclaimed Dividend"
    BANK_SAVINGS_DEPOSIT = "Inoperative Savings Deposit"
    BANK_FIXED_DEPOSIT = "Matured Fixed Deposit"
    MUTUAL_FUND_REDEMPTION = "Unclaimed Mutual Fund Redemption"
    INSURANCE_MATURITY = "Matured Life Insurance Policy"
    INSURANCE_SURVIVAL_BENEFIT = "Insurance Survival Benefit"
    CORPORATE_DEBENTURE = "Matured Corporate Debenture"


class ClaimStatusEnum(str, Enum):
    DISCOVERED = "Discovered / Unclaimed"
    DRAFT = "Draft"
    SUBMITTED = "Submitted to Regulators"
    UNDER_NODAL_REVIEW = "Under Nodal Officer / Bank Verification"
    APPROVED_PENDING_DISBURSAL = "Approved - In Disbursement Queue"
    DISBURSED = "Disbursed / Restituted to Demat & Bank"
    REJECTED = "Rejected / Discrepancy Flagged"


class ClaimantProfile(BaseModel):
    pan: str
    full_name: str
    date_of_birth: Optional[str] = None
    aadhaar_hash: Optional[str] = None
    mobile_number: Optional[str] = None
    email: Optional[str] = None
    digilocker_kyc_verified: bool = True
    bank_account_number: Optional[str] = None
    bank_ifsc: Optional[str] = None
    demat_account_id: Optional[str] = None


class UnclaimedAsset(BaseModel):
    asset_id: str
    regulator: RegulatorEnum
    entity_name: str
    asset_type: AssetTypeEnum
    account_or_folio_masked: str
    units_or_shares: Optional[int] = 0
    estimated_amount_inr: float
    dormancy_date: str
    holder_name: str
    confidence_score: float = 100.0
    is_legal_heir_claim: bool = False
    status: ClaimStatusEnum = ClaimStatusEnum.DISCOVERED


class SearchRequest(BaseModel):
    pan: str
    full_name: str
    date_of_birth: Optional[str] = None
    folio_or_account: Optional[str] = None


class SearchResponse(BaseModel):
    pan_searched: str
    claimant_name: str
    timestamp: datetime = Field(default_factory=utc_now)
    total_assets_found: int
    total_estimated_value_inr: float
    breakdown_by_regulator: Dict[str, Dict[str, Any]]
    assets: List[UnclaimedAsset]


class ClaimSubmissionRequest(BaseModel):
    claimant: ClaimantProfile
    selected_asset_ids: List[str]
    is_heir_succession: bool = False
    digilocker_document_ids: List[str] = Field(default_factory=list)
    indemnity_bond_signed: bool = True


class SubClaimStatus(BaseModel):
    sub_claim_id: str
    regulator: RegulatorEnum
    asset_id: str
    entity_name: str
    amount_inr: float
    status: ClaimStatusEnum
    statutory_sla_days: int
    submitted_at: datetime
    expected_completion_date: str
    last_log_message: str


class MasterClaimTrackingResponse(BaseModel):
    master_tracking_id: str
    claimant_pan: str
    claimant_name: str
    submitted_at: datetime
    overall_status: str
    total_restitution_value_inr: float
    sub_claims: List[SubClaimStatus]
    audit_trail: List[Dict[str, Any]]

