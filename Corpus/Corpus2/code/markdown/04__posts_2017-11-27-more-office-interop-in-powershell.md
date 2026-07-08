---
layout: post
title: More Office Interop in PowerShell
---

As part of our team's workflow we create various data files and then generate
tracking issues that we then import into our issue tracking system.

We have a semi-automated process to do this which works fairly well but for
some older issues we had imported I noticed that a vital piece of information
was missing.

When we ingest the issues into the system there is an identifier that we save
into the issue tracking system so we can find this information in our data
files later.

We also generate some reports from our data files one of which is an Excel
spreadsheet that contains the issue identifier and which also contains the
information that was missing from the issue tracking system.

Since there were hundreds of issue that needed updating I didn't want to update
all of the issues in the issue tracking system manually.

The issue tracking system allowed me to create a query and then download a CSV
of the issues that were missing the data. Then I found the spreadsheets that
had the data and wrote the following PowerShell script to generate a CSV file
with the missing data mapped to the issue identifiers:

```powershell
param(
    [Parameter(Mandatory)][string]$issuesCsv,
    [Parameter(Mandatory)][string]$excelReport
)

Add-Type -AssemblyName Microsoft.Office.Interop.Excel

function Get-IssueData {
    param(
        [Parameter(Mandatory)]$workbook,
        [Parameter(Mandatory)][PSCustomObject[]]$issues
    )

    $issueData = @()

    foreach ($issue in $issues) {
        if (-not $issue.IssueId) {
            continue
        }

        foreach ($worksheet in $workbook.Worksheets) {
            $target = $worksheet.UsedRange.Find($issueId)

            if ($target) {
                $csvIssue = [PSCustomObject]@{
                    IssueId = $issue.IssueId
                    MissingFieldData = $target.EntireRow.Value2[1, 5]
                }

                $issueData += $csvIssue

                break
            }
        }
    }

    return $issueData
}

try {
    $issues = Import-Csv -Path $path
} catch {
    "Unable to import issues."
    exit 1
}

$application = New-Object -ComObject Excel.Application

try {
    $workbook = $application.Workbooks.Open($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($excelReport))
} catch {
    "Unable to open workbook."
    $application.Quit()
    exit 1
}

Get-IssueData $workbook $issues | Export-Csv -Path export.csv -NoTypeInformation

$workbook.Close($false)

$application.Quit()
```
