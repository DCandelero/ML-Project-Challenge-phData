"""Unit tests for Demographics Service"""

import pytest
from ml.demographics_service import DemographicsService, ZipcodeNotFoundError


def test_demographics_service_loads_data():
    """Test demographics service loads CSV correctly"""
    service = DemographicsService("data/zipcode_demographics.csv")
    zipcodes = service.get_available_zipcodes()

    assert len(zipcodes) == 70  # Should load all 70 zipcodes
    assert '98115' in zipcodes  # Seattle zipcode
    assert '98042' in zipcodes  # Another Seattle area zipcode


def test_valid_zipcode_returns_demographics(demographics_service):
    """Test valid zipcode returns demographic data"""
    demographics = demographics_service.get_demographics('98115')

    assert isinstance(demographics, dict)
    assert 'ppltn_qty' in demographics
    assert 'medn_hshld_incm_amt' in demographics
    assert demographics['ppltn_qty'] > 0


def test_invalid_zipcode_raises_error(demographics_service):
    """Test invalid zipcode raises ZipcodeNotFoundError"""
    with pytest.raises(ZipcodeNotFoundError) as exc_info:
        demographics_service.get_demographics('99999')

    assert '99999' in str(exc_info.value)


def test_demographics_have_all_required_columns(demographics_service):
    """Test demographics contain all expected columns"""
    demographics = demographics_service.get_demographics('98115')

    required_columns = [
        'ppltn_qty', 'medn_hshld_incm_amt', 'hous_val_amt',
        'per_urbn', 'per_bchlr'
    ]

    for col in required_columns:
        assert col in demographics, f"Missing column: {col}"


def test_multiple_zipcode_lookups(demographics_service):
    """Test multiple zipcode lookups work correctly"""
    zipcode1 = demographics_service.get_demographics('98115')
    zipcode2 = demographics_service.get_demographics('98042')

    assert zipcode1 != zipcode2  # Different zipcodes should have different data
    assert zipcode1['ppltn_qty'] != zipcode2['ppltn_qty']


def test_get_available_zipcodes_returns_list(demographics_service):
    """Test get_available_zipcodes returns list"""
    zipcodes = demographics_service.get_available_zipcodes()

    assert isinstance(zipcodes, list)
    assert len(zipcodes) > 0
    assert all(isinstance(z, str) for z in zipcodes)
