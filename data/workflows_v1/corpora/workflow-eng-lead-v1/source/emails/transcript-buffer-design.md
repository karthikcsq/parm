Subject: Re: transcript buffer, streaming vs cap
From: me
To: Dan, Mei
Date: 2026-05-30

Dan's proposal is to make the transcript a ring buffer sized in entries. Mine
is to size it in bytes and spill the overflow to a temp file.

Entries are the wrong unit. A session that runs ten `grep` calls and one
`read` of a 40k-token file has eleven entries and almost all the memory is in
one of them. A byte cap bounds the thing we actually care about.

The spill file is the part worth arguing about. It means the transcript view
has to read from disk to scroll far back, which is slower. I think that is an
acceptable trade for not dying at four hours, and the alternative - dropping
old entries entirely - loses data the user can see in the UI.

Mei, if you disagree say so on the pull request rather than here so it is on
the record.
