from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from backend.models import (
    UnclaimedAsset, RegulatorEnum, AssetTypeEnum, ClaimStatusEnum, ClaimantProfile, SubClaimStatus
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BaseRegulatorAdapter:
    def __init__(self, regulator: RegulatorEnum):
        self.regulator = regulator

    def search(self, pan: str, name: str, dob: Optional[str] = None, folio: Optional[str] = None) -> List[UnclaimedAsset]:
        raise NotImplementedError

    def submit_claim(self, claimant: ClaimantProfile, asset: UnclaimedAsset, is_heir: bool) -> SubClaimStatus:
        raise NotImplementedError

    def get_claim_status(self, sub_claim_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class IEPFAAdapter(BaseRegulatorAdapter):
    def __init__(self):
        super().__init__(RegulatorEnum.IEPFA)
        self.mock_registry = [
            {
                "asset_id": "IEPF-SHR-2024-8891",
                "entity_name": "Reliance Industries Limited",
                "asset_type": AssetTypeEnum.EQUITY_SHARES,
                "account_or_folio_masked": "FOLIO-****9201",
                "units_or_shares": 150,
                "estimated_amount_inr": 435000.0,
                "dormancy_date": "2016-11-15",
                "holder_name": "SOUMYA BANTHIA",
                "pan": "ABCDE1234F"
            },
            {
                "asset_id": "IEPF-DIV-2023-4102",
                "entity_name": "Tata Consultancy Services Ltd",
                "asset_type": AssetTypeEnum.UNCLAIMED_DIVIDEND,
                "account_or_folio_masked": "FOLIO-****4412",
                "units_or_shares": 0,
                "estimated_amount_inr": 18450.0,
                "dormancy_date": "2017-08-20",
                "holder_name": "SOUMYA BANTHIA",
                "pan": "ABCDE1234F"
            },
            {
                "asset_id": "IEPF-SHR-2022-1094",
                "entity_name": "ITC Limited",
                "asset_type": AssetTypeEnum.EQUITY_SHARES,
                "account_or_folio_masked": "FOLIO-****0034",
                "units_or_shares": 500,
                "estimated_amount_inr": 215000.0,
                "dormancy_date": "2015-09-30",
                "holder_name": "RAMESH BANTHIA",
                "pan": "WXYZR9876Q"
            }
        ]

    def search(self, pan: str, name: str, dob: Optional[str] = None, folio: Optional[str] = None) -> List[UnclaimedAsset]:
        results = []
        pan_clean = pan.strip().upper()
        name_clean = name.strip().upper()

        for item in self.mock_registry:
            if item["pan"] == pan_clean or name_clean in item["holder_name"] or item["holder_name"] in name_clean:
                is_heir = item["pan"] != pan_clean
                results.append(
                    UnclaimedAsset(
                        asset_id=item["asset_id"],
                        regulator=self.regulator,
                        entity_name=item["entity_name"],
                        asset_type=item["asset_type"],
                        account_or_folio_masked=item["account_or_folio_masked"],
                        units_or_shares=item["units_or_shares"],
                        estimated_amount_inr=item["estimated_amount_inr"],
                        dormancy_date=item["dormancy_date"],
                        holder_name=item["holder_name"],
                        confidence_score=100.0 if item["pan"] == pan_clean else 85.0,
                        is_legal_heir_claim=is_heir,
                        status=ClaimStatusEnum.DISCOVERED
                    )
                )
        return results

    def submit_claim(self, claimant: ClaimantProfile, asset: UnclaimedAsset, is_heir: bool) -> SubClaimStatus:
        sub_id = f"MCA-IEPF-CLM-{utc_now().strftime('%Y%m%d%H%M%S')}"
        return SubClaimStatus(
            sub_claim_id=sub_id,
            regulator=self.regulator,
            asset_id=asset.asset_id,
            entity_name=asset.entity_name,
            amount_inr=asset.estimated_amount_inr,
            status=ClaimStatusEnum.SUBMITTED,
            statutory_sla_days=30,
            submitted_at=utc_now(),
            expected_completion_date="Within 30 business days",
            last_log_message="Form IEPF-5 routed digitally to Company Nodal Officer with DigiLocker verified KYC."
        )


class RBIUDGAMAdapter(BaseRegulatorAdapter):
    def __init__(self):
        super().__init__(RegulatorEnum.RBI_UDGAM)
        self.mock_registry = [
            {
                "asset_id": "RBI-DEAF-SB-88219",
                "entity_name": "State Bank of India (SBI)",
                "asset_type": AssetTypeEnum.BANK_SAVINGS_DEPOSIT,
                "account_or_folio_masked": "SB-****8102",
                "units_or_shares": 0,
                "estimated_amount_inr": 34800.0,
                "dormancy_date": "2014-03-31",
                "holder_name": "SOUMYA BANTHIA",
                "pan": "ABCDE1234F"
            },
            {
                "asset_id": "RBI-DEAF-FD-90124",
                "entity_name": "Punjab National Bank (PNB)",
                "asset_type": AssetTypeEnum.BANK_FIXED_DEPOSIT,
                "account_or_folio_masked": "FD-****1190",
                "units_or_shares": 0,
                "estimated_amount_inr": 125000.0,
                "dormancy_date": "2013-06-15",
                "holder_name": "SOUMYA BANTHIA",
                "pan": "ABCDE1234F"
            }
        ]

    def search(self, pan: str, name: str, dob: Optional[str] = None, folio: Optional[str] = None) -> List[UnclaimedAsset]:
        results = []
        pan_clean = pan.strip().upper()
        name_clean = name.strip().upper()

        for item in self.mock_registry:
            if item["pan"] == pan_clean or name_clean in item["holder_name"]:
                results.append(
                    UnclaimedAsset(
                        asset_id=item["asset_id"],
                        regulator=self.regulator,
                        entity_name=item["entity_name"],
                        asset_type=item["asset_type"],
                        account_or_folio_masked=item["account_or_folio_masked"],
                        units_or_shares=0,
                        estimated_amount_inr=item["estimated_amount_inr"],
                        dormancy_date=item["dormancy_date"],
                        holder_name=item["holder_name"],
                        confidence_score=100.0 if item["pan"] == pan_clean else 88.0,
                        is_legal_heir_claim=False,
                        status=ClaimStatusEnum.DISCOVERED
                    )
                )
        return results

    def submit_claim(self, claimant: ClaimantProfile, asset: UnclaimedAsset, is_heir: bool) -> SubClaimStatus:
        sub_id = f"RBI-UDGAM-CLM-{utc_now().strftime('%Y%m%d%H%M%S')}"
        return SubClaimStatus(
            sub_claim_id=sub_id,
            regulator=self.regulator,
            asset_id=asset.asset_id,
            entity_name=asset.entity_name,
            amount_inr=asset.estimated_amount_inr,
            status=ClaimStatusEnum.SUBMITTED,
            statutory_sla_days=15,
            submitted_at=utc_now(),
            expected_completion_date="Within 15 business days",
            last_log_message="Claim dispatched to Bank Nodal Branch. Direct e-Kuber refund initiated upon bank verification."
        )


class SEBIAdapter(BaseRegulatorAdapter):
    def __init__(self):
        super().__init__(RegulatorEnum.SEBI)
        self.mock_registry = [
            {
                "asset_id": "SEBI-MF-CAMS-7712",
                "entity_name": "HDFC Mutual Fund (CAMS)",
                "asset_type": AssetTypeEnum.MUTUAL_FUND_REDEMPTION,
                "account_or_folio_masked": "MF-****3309",
                "units_or_shares": 120,
                "estimated_amount_inr": 52400.0,
                "dormancy_date": "2018-01-10",
                "holder_name": "SOUMYA BANTHIA",
                "pan": "ABCDE1234F"
            }
        ]

    def search(self, pan: str, name: str, dob: Optional[str] = None, folio: Optional[str] = None) -> List[UnclaimedAsset]:
        results = []
        pan_clean = pan.strip().upper()
        for item in self.mock_registry:
            if item["pan"] == pan_clean:
                results.append(
                    UnclaimedAsset(
                        asset_id=item["asset_id"],
                        regulator=self.regulator,
                        entity_name=item["entity_name"],
                        asset_type=item["asset_type"],
                        account_or_folio_masked=item["account_or_folio_masked"],
                        units_or_shares=item["units_or_shares"],
                        estimated_amount_inr=item["estimated_amount_inr"],
                        dormancy_date=item["dormancy_date"],
                        holder_name=item["holder_name"],
                        confidence_score=100.0,
                        is_legal_heir_claim=False,
                        status=ClaimStatusEnum.DISCOVERED
                    )
                )
        return results

    def submit_claim(self, claimant: ClaimantProfile, asset: UnclaimedAsset, is_heir: bool) -> SubClaimStatus:
        sub_id = f"SEBI-AMFI-CLM-{utc_now().strftime('%Y%m%d%H%M%S')}"
        return SubClaimStatus(
            sub_claim_id=sub_id,
            regulator=self.regulator,
            asset_id=asset.asset_id,
            entity_name=asset.entity_name,
            amount_inr=asset.estimated_amount_inr,
            status=ClaimStatusEnum.SUBMITTED,
            statutory_sla_days=10,
            submitted_at=utc_now(),
            expected_completion_date="Within 10 business days",
            last_log_message="Claim transmitted to Mutual Fund RTA for direct bank credit via penny-drop validation."
        )


class IRDAIAdapter(BaseRegulatorAdapter):
    def __init__(self):
        super().__init__(RegulatorEnum.IRDAI)
        self.mock_registry = [
            {
                "asset_id": "IRDAI-LIC-POL-55410",
                "entity_name": "Life Insurance Corporation of India (LIC)",
                "asset_type": AssetTypeEnum.INSURANCE_MATURITY,
                "account_or_folio_masked": "POL-****6612",
                "units_or_shares": 0,
                "estimated_amount_inr": 85000.0,
                "dormancy_date": "2015-12-01",
                "holder_name": "SOUMYA BANTHIA",
                "pan": "ABCDE1234F"
            }
        ]

    def search(self, pan: str, name: str, dob: Optional[str] = None, folio: Optional[str] = None) -> List[UnclaimedAsset]:
        results = []
        pan_clean = pan.strip().upper()
        for item in self.mock_registry:
            if item["pan"] == pan_clean:
                results.append(
                    UnclaimedAsset(
                        asset_id=item["asset_id"],
                        regulator=self.regulator,
                        entity_name=item["entity_name"],
                        asset_type=item["asset_type"],
                        account_or_folio_masked=item["account_or_folio_masked"],
                        units_or_shares=0,
                        estimated_amount_inr=item["estimated_amount_inr"],
                        dormancy_date=item["dormancy_date"],
                        holder_name=item["holder_name"],
                        confidence_score=100.0,
                        is_legal_heir_claim=False,
                        status=ClaimStatusEnum.DISCOVERED
                    )
                )
        return results

    def submit_claim(self, claimant: ClaimantProfile, asset: UnclaimedAsset, is_heir: bool) -> SubClaimStatus:
        sub_id = f"IRDAI-INS-CLM-{utc_now().strftime('%Y%m%d%H%M%S')}"
        return SubClaimStatus(
            sub_claim_id=sub_id,
            regulator=self.regulator,
            asset_id=asset.asset_id,
            entity_name=asset.entity_name,
            amount_inr=asset.estimated_amount_inr,
            status=ClaimStatusEnum.SUBMITTED,
            statutory_sla_days=21,
            submitted_at=utc_now(),
            expected_completion_date="Within 21 business days",
            last_log_message="Claim received at Insurer Central Processing Unit for policy discharge."
        )
