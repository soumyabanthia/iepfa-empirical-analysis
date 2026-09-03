from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    SearchRequest, SearchResponse, ClaimSubmissionRequest,
    MasterClaimTrackingResponse, SubClaimStatus, RegulatorEnum,
    UnclaimedAsset
)
from backend.adapters import IEPFAAdapter, RBIUDGAMAdapter, SEBIAdapter, IRDAIAdapter


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


app = FastAPI(title="Unclaimed Assets Gateway API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

adapters = {
    RegulatorEnum.IEPFA: IEPFAAdapter(),
    RegulatorEnum.RBI_UDGAM: RBIUDGAMAdapter(),
    RegulatorEnum.SEBI: SEBIAdapter(),
    RegulatorEnum.IRDAI: IRDAIAdapter()
}

CLAIMS_DB: Dict[str, MasterClaimTrackingResponse] = {}


@app.get("/")
def root():
    return {
        "portal": "Unified National Unclaimed Financial Assets Gateway (NARG)",
        "status": "Operational",
        "regulators_connected": [reg.value for reg in adapters.keys()],
        "version": "1.0.0"
    }


@app.post("/api/v1/search", response_model=SearchResponse)
def unified_search(request: SearchRequest):
    all_assets: List[UnclaimedAsset] = []
    regulator_breakdown: Dict[str, Dict[str, Any]] = {}

    for reg_enum, adapter in adapters.items():
        found = adapter.search(
            pan=request.pan,
            name=request.full_name,
            dob=request.date_of_birth,
            folio=request.folio_or_account
        )
        total_val = sum(a.estimated_amount_inr for a in found)
        regulator_breakdown[reg_enum.value] = {
            "count": len(found),
            "estimated_value_inr": round(total_val, 2)
        }
        all_assets.extend(found)

    total_val_all = sum(a.estimated_amount_inr for a in all_assets)

    return SearchResponse(
        pan_searched=request.pan.upper(),
        claimant_name=request.full_name.upper(),
        timestamp=utc_now(),
        total_assets_found=len(all_assets),
        total_estimated_value_inr=round(total_val_all, 2),
        breakdown_by_regulator=regulator_breakdown,
        assets=all_assets
    )


@app.post("/api/v1/claims/submit", response_model=MasterClaimTrackingResponse)
def submit_unified_claim(request: ClaimSubmissionRequest):
    if not request.selected_asset_ids:
        raise HTTPException(status_code=400, detail="No assets selected for claim submission.")

    selected_assets: List[UnclaimedAsset] = []
    for reg_enum, adapter in adapters.items():
        search_res = adapter.search(request.claimant.pan, request.claimant.full_name)
        for a in search_res:
            if a.asset_id in request.selected_asset_ids:
                selected_assets.append(a)

    if not selected_assets:
        raise HTTPException(status_code=404, detail="Selected asset IDs could not be validated.")

    master_id = f"NAT-ASSET-REC-{utc_now().strftime('%Y%m%d%H%M%S')}"
    sub_claims: List[SubClaimStatus] = []

    for asset in selected_assets:
        adapter = adapters[asset.regulator]
        sub_status = adapter.submit_claim(
            claimant=request.claimant,
            asset=asset,
            is_heir=request.is_heir_succession
        )
        sub_claims.append(sub_status)

    total_val = sum(a.estimated_amount_inr for a in selected_assets)

    audit_entry = {
        "timestamp": utc_now().isoformat(),
        "action": "UNIFIED_CLAIM_SUBMITTED",
        "description": f"Successfully routed {len(sub_claims)} claims to {len(set(a.regulator for a in selected_assets))} regulatory nodes.",
        "digilocker_verified": request.claimant.digilocker_kyc_verified
    }

    claim_record = MasterClaimTrackingResponse(
        master_tracking_id=master_id,
        claimant_pan=request.claimant.pan.upper(),
        claimant_name=request.claimant.full_name.upper(),
        submitted_at=utc_now(),
        overall_status="IN_PROGRESS_WITH_REGULATORS",
        total_restitution_value_inr=round(total_val, 2),
        sub_claims=sub_claims,
        audit_trail=[audit_entry]
    )

    CLAIMS_DB[master_id] = claim_record
    return claim_record


@app.get("/api/v1/claims/{master_id}/track", response_model=MasterClaimTrackingResponse)
def track_claim_lifecycle(master_id: str):
    if master_id not in CLAIMS_DB:
        raise HTTPException(status_code=404, detail=f"Claim Tracking ID '{master_id}' not found.")
    return CLAIMS_DB[master_id]


@app.get("/api/v1/analytics/summary")
def get_systemic_analytics():
    return {
        "national_unclaimed_corpus_inr_cr": 108470.0,
        "breakdown_by_authority": {
            "IEPFA (MCA)": {"corpus_inr_cr": 38500.0, "avg_sla_days": 42, "annual_growth_pct": 14.5},
            "RBI-UDGAM (DEAF)": {"corpus_inr_cr": 42270.0, "avg_sla_days": 18, "annual_growth_pct": 18.2},
            "SEBI (Mutual Funds/IPF)": {"corpus_inr_cr": 3200.0, "avg_sla_days": 12, "annual_growth_pct": 8.7},
            "IRDAI (Unclaimed Insurance)": {"corpus_inr_cr": 24500.0, "avg_sla_days": 26, "annual_growth_pct": 12.0}
        },
        "estimated_annual_deadweight_loss_cr": 8677.6,
        "digitization_fast_track_potential_reduction_pct": 65.0
    }

