
const express = require('express')
const search = require('./predict')

const app = express()
const port = process.env.PORT || 3000

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html')
})

app.get('/similar/:word', async (req, res) => {
    const {word} = req.params
    const ret = await search(word)
    if (!ret) {
      res.send({
        ok: false,
        message: 'Invalid word',
      })
      return
    }
    res.send({
      ok: true,
      data: ret,
    })
})

app.listen(port, () => {
  // search('nice')
  console.log(`SemanticIcon listening at http://localhost:${port}`)
})
