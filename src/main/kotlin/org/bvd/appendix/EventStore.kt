package org.bvd.appendix

import com.fasterxml.jackson.databind.ObjectMapper
import java.sql.Connection
import java.time.Instant
import java.util.*

class EventStore(val mapper: ObjectMapper) {
    private companion object {
        const val INSERT_EVENTS_SQL: String =
            "INSERT INTO events (event_id, stream_id, version, event_type, payload, emitted_at) VALUES (?, ?, ?, ?, ?, ?)"
        const val READ_EVENTS_SQL: String =
            "SELECT event_id, stream_id, version, event_type, payload, emitted_at FROM events WHERE stream_id = ? ORDER BY version ASC"
    }

    fun append(
        connection: Connection,
        events: List<Any>,
        streamId: UUID,
        expectedVersion: Long
    ) {
        if (events.isEmpty()) {
            return
        }

        connection.prepareStatement(INSERT_EVENTS_SQL).use { statement ->

            for (event in events) {
                statement.setObject(1, UUID.randomUUID())
                statement.setObject(2, streamId)
                statement.setLong(3, expectedVersion + 1)
                statement.setString(4, event::class.qualifiedName)
                statement.setBytes(5, mapper.writeValueAsBytes(event))
                statement.setObject(6, Instant.now())
                statement.addBatch()
            }
        }
    }

    fun loadEvents(connection: Connection, streamId: UUID): List<Event> {
        connection.prepareStatement(READ_EVENTS_SQL).use { statement ->
            statement.setObject(1, streamId)

            statement.executeQuery().use { resultSet ->
                val events = mutableListOf<Event>()

                while (resultSet.next()) {
                    val eventType = resultSet.getString("event_type")
                    val payloadBytes: ByteArray = resultSet.getBytes("payload")

                    val payloadClass = Class.forName(eventType)

                    events.add(mapper.readValue(payloadBytes, payloadClass) as Event)
                }

                return events
            }
        }
    }
}
