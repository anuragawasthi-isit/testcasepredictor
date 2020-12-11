import json
import sys
import html2text
from tfs import TFSAPI
import pandas as pd
from dateutil import parser


def getactnccrit():
    # HTTP Basic Auth with personal access token
    client = TFSAPI("http://tfs2.dell.com:8080/tfs/eDell/", project="eDellPrograms",
                    pat='dgi5cehn7jg3rpn5enmovn2xj47thnx77vsg5uyv7wiz227ckpaa')
    query = client.run_query('51bc4c75-3482-46c5-8579-8ebd719bc441')
    # Get all found workitems
    result = query.result
    workitems = result.workitems

    for wrktimid in workitems:
        wrkitmfields = wrktimid.fields
        acptnccrit = wrktimid['Microsoft.VSTS.Common.AcceptanceCriteria']
        clean_text = html2text.html2text(acptnccrit)
        print(clean_text)


def gettcsteps():
    # HTTP Basic Auth with personal access token
    client = TFSAPI("http://tfs2.dell.com:8080/tfs/eDell/", project="eDellPrograms",
                    pat='dgi5cehn7jg3rpn5enmovn2xj47thnx77vsg5uyv7wiz227ckpaa')
    query = client.run_query('a7391bf8-addc-46a8-b5a8-0823cb777637')
    # Get all found workitems
    result = query.result
    workitems = result.workitems
    openwith = {}
    lstkeys = []
    lstvalues = []

    for wrktimid in workitems:
        wrkitmfields = wrktimid.fields
        tcid = wrktimid['system.id']
        tctitle = wrktimid['system.title']
        tcsteps = wrktimid['microsoft.vsts.tcm.steps']
        if tcsteps is not None:
            clean_text = html2text.html2text(tcsteps)
            cleaner_step = html2text.html2text(clean_text).replace("\n"," ")

        openwith[tcid] = [str(tcid), str(tctitle), str(cleaner_step)]

    for key, value in openwith.items():
        print(key, ' --> ', value)
        lstkeys.append(key)
        lstvalues.append(value)

    dfObj = pd.DataFrame(lstvalues, columns=['id', 'tctitle', 'tcsteps'])
    dfObj.to_csv('dellcom.csv',index=False)


def getquerydetails():
    # HTTP Basic Auth with personal access token
    client = TFSAPI("http://tfs2.dell.com:8080/tfs/eDell/", project="eDellPrograms",
                    pat='dgi5cehn7jg3rpn5enmovn2xj47thnx77vsg5uyv7wiz227ckpaa')
    query = client.run_query('c03cebf1-1bb2-4c8a-9e75-76ba65ef4020')
    # Get all found workitems
    result = query.result
    workitems = result.workItemRelations
    a = workitems[0].target
    wrkitemlist = []
    lstkeys = []
    lstvalues = []
    openwith = {}
    for wrktimid in workitems:
        trgt = wrktimid.target
        wrkitemlist.append(int(trgt.id))

    for b in workitems:
        strchanges = []
        stre2estrtchanges = []
        trgt = b.target
        sit_strt = False
        e2e_strt = False
        ttltimeelapsed = ''
        applicatn = ''
        # expctdappln =sys.argv[1]
        wrkitm = client.get_workitem(int(trgt.id))
        wrkitmfields = wrkitm.fields
        if ('Dell.SDLC.Application' in wrkitmfields.keys()):
            applicatn = wrkitmfields['Dell.SDLC.Application']

        if ('OSC' in applicatn):
            if (str(wrkitm['System.WorkItemType']) == "Story" or str(wrkitm['System.WorkItemType']) == "Feature"):
                rev = wrkitm.revisions
                for re in rev:
                    fieldacss = re.data
                    for key, value in fieldacss.items():
                        if (key == 'fields'):
                            allfields = fieldacss[key]
                            if ('Dell.SDLC.Application' in allfields.keys()):
                                applicatn = fieldacss[key]['Dell.SDLC.Application']

                            revstate = fieldacss[key]['System.State']
                            strchngdby = fieldacss[key]['System.ChangedBy']
                            # stre2ereqd = fieldacss[key]['Dell.SDLC.QERequired']

                            if ('OSC' in applicatn):
                                if (revstate == 'Ready for E2E testing'):
                                    sit_strt = True
                                    changedate = fieldacss[key]['System.ChangedDate']
                                    if (trgt.id in wrkitemlist):
                                        wrkitemlist.remove(trgt.id)

                                    stre2estrtchanges.append(changedate)
                                    # strchanges.append(strchngdby)
                                    # strchanges.append(stre2ereqd)
                                if (revstate == 'E2E Testing Started'):
                                    e2e_strt = True
                                    e2echangedate = fieldacss[key]['System.ChangedDate']
                                    if (trgt.id in wrkitemlist):
                                        wrkitemlist.remove(trgt.id)
                                    strchanges.append(e2echangedate)
                                    strchanges.append(strchngdby)
                                    # strchanges.append(stre2ereqd)
                            if (sit_strt == True and e2e_strt == True):
                                e2eprpdate = parser.parse(stre2estrtchanges[0])
                                e2estrtpdate = parser.parse(strchanges[-2])
                                ttltimeelapsed = e2estrtpdate - e2eprpdate
                                openwith[trgt.id] = [str(trgt.id), str(ttltimeelapsed.days), stre2estrtchanges[0],
                                                     strchanges[-2], strchanges[-1]]

    for key, value in openwith.items():
        print(key, ' --> ', value)
        lstkeys.append(key)
        lstvalues.append(value)

    dfObj = pd.DataFrame(lstvalues, columns=['id', 'totaltime', 'readyfor2e', 'e2estarted', 'owner'])
    # name_dict = {
    #     'Name': lstkeys,
    #     'changeby': lstvalues[0][2],
    #     'totaltimeelapsed': lstvalues[0]
    # }
    dfObj.to_csv('OSC.csv')
    # df1 =pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in name_dict.items() ]))
    # df = pd.DataFrame(name_dict)
    # df1.to_csv('osc.csv')


# getquerydetails()
# getactnccrit()
gettcsteps()
