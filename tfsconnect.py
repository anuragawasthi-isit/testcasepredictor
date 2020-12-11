import json

import html2text
from tfs import TFSAPI


def getworkitemtitle(wrkitemid):
    # HTTP Basic Auth with personal access token
    client = TFSAPI("TFS URL HERE", project="TFS PROJECT",
                    pat='patid')
    # For single Workitem
    workitem = client.get_workitem(int(wrkitemid))

    # Get all fields
    # Case insensitive. Remove space in field name
    return workitem['Title']


def getacptnccriteria(wrkitemid):
    # HTTP Basic Auth with personal access token
    client = TFSAPI("TFS URL HERE", project="TFS PROJECT",
                    pat='patid')
    # For single Workitem

    workitem = client.get_workitem(int(wrkitemid))
    actnccrit = html2text.html2text(workitem['Microsoft.VSTS.Common.AcceptanceCriteria'])
    # Get all fields
    # Case insensitive. Remove space in field name
    return actnccrit






# Run New query 1 in Shared Queries folder
# query = client.run_query('Shared Queries/FY21-0602 Seller-Total Scope')
# You can also use query GUID
# query = client.run_query('c03cebf1-1bb2-4c8a-9e75-76ba65ef4020')
